import asyncio
from itertools import chain
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
)

from aiostream import stream as async_stream

from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (
    ActionReasoningStep,
    ActionReasoningStepArr,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.agent.types import BaseAgent
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.memory.types import BaseMemory
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.tools.types import AsyncBaseTool
from llama_index.utils import async_unit_generator, print_text, unit_generator

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


class ReActAgent(BaseAgent):
    """ReAct agent.

    Uses a ReAct prompt that can be used in both chat and text
    completion endpoints.

    Can take in a set of tools that require structured inputs.
    """

    def __init__(
        self,
        retrieve_tool,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        super().__init__(callback_manager=callback_manager or llm.callback_manager)
        self._llm = llm
        self._memory = memory
        self._max_iterations = max_iterations
        self._react_chat_formatter = react_chat_formatter or ReActChatFormatter()
        self._output_parser = output_parser or ReActOutputParser()
        self._verbose = verbose
        self.sources: List[ToolOutput] = []
        self._retrieve_tool = retrieve_tool

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        retrieve_tool,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ReActAgent":
        """Convenience constructor method from set of of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgent
        """
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        memory = memory or memory_cls.from_defaults(
            chat_history=chat_history or [], llm=llm
        )
        return cls(
            retrieve_tool = retrieve_tool,
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Chat history."""
        return self._memory.get_all()

    def reset(self) -> None:
        self._memory.reset()

    def _extract_reasoning_step(
        self, output: ChatResponse, is_streaming: bool = False
    ) -> Tuple[str, List[BaseReasoningStep], bool]:
        """
        Extracts the reasoning step from the given output.

        This method parses the message content from the output,
        extracts the reasoning step, and determines whether the processing is
        complete. It also performs validation checks on the output and
        handles possible errors.
        """
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content, is_streaming)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        current_reasoning.append(reasoning_step)

        if reasoning_step.is_done:
            return message_content, current_reasoning, True

        reasoning_step = cast(ActionReasoningStepArr, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStepArr):
            raise ValueError(f"Expected ActionReasoningStepArr, got {reasoning_step}")

        return message_content, current_reasoning, False

    def _process_actions(
        self,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict: Dict[str, AsyncBaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        _, current_reasoning, is_done = self._extract_reasoning_step(
            output, is_streaming
        )

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = tool.call(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        self.sources.append(tool_output)

        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False
    
    async def _atool_request(self, tools_dict, reasoning_step, i):
        tool = tools_dict[reasoning_step.actions[i]]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_inputs[i],
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = await tool.acall(**reasoning_step.action_inputs[i])
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        self.sources.append(tool_output)
        #observation_step = ObservationReasoningStep(observation=str(tool_output))
        #print("observation_responce: ", str(tool_output), type(tool_output))
        return str(tool_output)

    async def _aprocess_actions(
        self,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict = {tool.metadata.name: tool for tool in tools}
        _, current_reasoning, is_done = self._extract_reasoning_step(
            output, is_streaming
        )

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStepArr, current_reasoning[-1])
        #print("reasoning_step: ", reasoning_step)

        #for i in range(len(reasoning_step.actions)):
        #    print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii: ", i)
            #observation_responce = await self._atool_request(tools_dict, reasoning_step, i)
        steps = []
        for i in range(len(reasoning_step.actions)):
            steps.append(self._atool_request(tools_dict, reasoning_step, i))

        group = asyncio.gather(*steps
                #current_reasoning.append(observation_responce)
        )

        #print("group-----------------------------------------------------------: ", group)
        group_ = await group
        observation_ = ""
        for idx, i in enumerate(group_):
            observation_ = observation_ + "\n" + "Observation " + str(idx) + "\n" + str(i)
        #print("observation_: ",observation_)
        observation_step = ObservationReasoningStep(observation=str(observation_))
        current_reasoning.append(observation_step)
        
        #print("observation_responce: ", observation_responce)
        #current_reasoning.append(observation_responce)
        #if self._verbose:
        #    print_text(f"{observation_responce.get_content()}\n", color="blue")
        return current_reasoning, False

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        print(current_reasoning[-1])
        response_step = cast(ResponseReasoningStep, current_reasoning[-1])

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_step.response, sources=self.sources)

    def _infer_stream_chunk_is_final(self, chunk: ChatResponse) -> bool:
        """Infers if a chunk from a live stream is the start of the final
        reasoning step. (i.e., and should eventually become
        ResponseReasoningStep — not part of this function's logic tho.).

        Args:
            chunk (ChatResponse): the current chunk stream to check

        Returns:
            bool: Boolean on whether the chunk is the start of the final response
        """
        latest_content = chunk.message.content
        if latest_content:
            if not latest_content.startswith(
                "Thought"
            ):  # doesn't follow thought-action format
                return True
            else:
                if "Answer: " in latest_content:
                    return True
        return False

    def _add_back_chunk_to_stream(
        self, chunk: ChatResponse, chat_stream: Generator[ChatResponse, None, None]
    ) -> Generator[ChatResponse, None, None]:
        """Helper method for adding back initial chunk stream of final response
        back to the rest of the chat_stream.

        Args:
            chunk (ChatResponse): the chunk to add back to the beginning of the
                                    chat_stream.

        Return:
            Generator[ChatResponse, None, None]: the updated chat_stream
        """
        updated_stream = chain.from_iterable(  # need to add back partial response chunk
            [
                unit_generator(chunk),
                chat_stream,
            ]
        )
        # use cast to avoid mypy issue with chain and Generator
        updated_stream_c: Generator[ChatResponse, None, None] = cast(
            Generator[ChatResponse, None, None], updated_stream
        )
        return updated_stream_c

    def _async_add_back_chunk_to_stream(
        self, chunk: ChatResponse, chat_stream: AsyncGenerator[ChatResponse, None]
    ) -> AsyncGenerator[ChatResponse, None]:
        """Helper method for adding back initial chunk stream of final response
        back to the rest of the chat_stream.

        NOTE: this itself is not an async function.

        Args:
            chunk (ChatResponse): the chunk to add back to the beginning of the
                                    chat_stream.

        Return:
            AsyncGenerator[ChatResponse, None]: the updated async chat_stream
        """
        updated_stream = (
            async_stream.combine.merge(  # need to add back partial response chunk
                async_unit_generator(chunk),
                chat_stream,
            )
        )
        # use cast to avoid mypy issue with Stream and AsyncGenerator
        updated_stream_c: AsyncGenerator[ChatResponse, None] = cast(
            AsyncGenerator[ChatResponse, None], updated_stream
        )
        return updated_stream_c

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        # get tools
        # TODO: do get tools dynamically at every iteration of the agent loop
        self.sources = []
        tools = self.get_tools(message)

        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                tools,
                chat_history=self._memory.get(),
                current_reasoning=current_reasoning,
            )
            # send prompt
            chat_response = self._llm.chat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(
                tools, output=chat_response
            )
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        # get tools
        # TODO: do get tools dynamically at every iteration of the agent loop
        self.sources = []
        tools = self.get_tools(message)

        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                tools,
                chat_history=self._memory.get(),
                current_reasoning=current_reasoning,
            )
            #print("input_chat: ", input_chat)
            # send prompt
            chat_response = await self._llm.achat(input_chat)
            #print("chat_response: ", chat_response)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = await self._aprocess_actions(
                tools, output=chat_response
            )
            #loop = asyncio.get_event_loop()
            #try:
            #    loop.run_until_complete(self._aprocess_actions(tools, output=chat_response))
            #    loop.run_until_complete(loop.shutdown_asyncgens())
            #finally:
            #    loop.close()
            #print("reasoning_steps:", reasoning_steps, type(reasoning_steps))
            current_reasoning.extend(reasoning_steps)
            #print("current_reasoning:", current_reasoning, len(current_reasoning))
            if is_done:
                break
        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        # get tools
        # TODO: do get tools dynamically at every iteration of the agent loop
        self.sources = []
        tools = self.get_tools(message)

        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        is_done, ix = False, 0
        while (not is_done) and (ix < self._max_iterations):
            ix += 1

            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                tools,
                chat_history=self._memory.get(),
                current_reasoning=current_reasoning,
            )
            # send prompt
            chat_stream = self._llm.stream_chat(input_chat)

            # iterate over stream, break out if is final answer after the "Answer: "
            full_response = ChatResponse(
                message=ChatMessage(content=None, role="assistant")
            )
            for latest_chunk in chat_stream:
                full_response = latest_chunk
                is_done = self._infer_stream_chunk_is_final(latest_chunk)
                if is_done:
                    break

            # given react prompt outputs, call tools or return response
            reasoning_steps, _ = self._process_actions(
                tools=tools, output=full_response, is_streaming=True
            )
            current_reasoning.extend(reasoning_steps)

        # Get the response in a separate thread so we can yield the response
        response_stream = self._add_back_chunk_to_stream(
            chunk=latest_chunk, chat_stream=chat_stream
        )

        chat_stream_response = StreamingAgentChatResponse(
            chat_stream=response_stream,
            sources=self.sources,
        )
        thread = Thread(
            target=chat_stream_response.write_response_to_history,
            args=(self._memory,),
        )
        thread.start()
        return chat_stream_response

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        # get tools
        # TODO: do get tools dynamically at every iteration of the agent loop
        self.sources = []
        tools = self.get_tools(message)

        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        is_done, ix = False, 0
        while (not is_done) and (ix < self._max_iterations):
            ix += 1

            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                tools,
                chat_history=self._memory.get(),
                current_reasoning=current_reasoning,
            )
            # send prompt
            chat_stream = await self._llm.astream_chat(input_chat)

            # iterate over stream, break out if is final answer
            is_done = False
            full_response = ChatResponse(
                message=ChatMessage(content=None, role="assistant")
            )
            async for latest_chunk in chat_stream:
                full_response = latest_chunk
                is_done = self._infer_stream_chunk_is_final(latest_chunk)
                if is_done:
                    break

            # given react prompt outputs, call tools or return response
            reasoning_steps, _ = self._process_actions(
                tools=tools, output=full_response, is_streaming=True
            )
            current_reasoning.extend(reasoning_steps)

        # Get the response in a separate thread so we can yield the response
        response_stream = self._async_add_back_chunk_to_stream(
            chunk=latest_chunk, chat_stream=chat_stream
        )

        chat_stream_response = StreamingAgentChatResponse(
            achat_stream=response_stream, sources=self.sources
        )
        # create task to write chat response to history
        asyncio.create_task(
            chat_stream_response.awrite_response_to_history(self._memory)
        )
        # thread.start()
        return chat_stream_response

    #def get_tools(self, message: str) -> List[AsyncBaseTool]:
    #    """Get tools."""
    #    return [adapt_to_async_tool(t) for t in self._get_tools(message)]

    def get_tools(self, message: str) -> List[AsyncBaseTool]:
        """Get tools."""
        tools = [adapt_to_async_tool(t) for t in self._get_tools(message)]
        ##print("tools: ", tools)
        #docs = [str("idx: " + str(idx) + ", name: " + str(t.metadata.name) + ", description: " + str(t.metadata.description)) for idx, t in enumerate(tools)]
        ##print("docs: ", docs)
        #from llama_index import Document, VectorStoreIndex
        #documents = [Document(text=t, metadata = {"idx": idx}) for idx, t in enumerate(docs)]
#
        #from llama_index.embeddings import OpenAIEmbedding
#
        #embed_model = OpenAIEmbedding()
#
        #index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#
        #from llama_index.retrievers import VectorIndexRetriever
#
        #retriever = VectorIndexRetriever(
        #    index=index,
        #    similarity_top_k=5,
        #)

        response = self._retrieve_tool.retrieve(message)

        #print("response: ", response)

        tools_ = []
        #for val in response.metadata.values():
        #    tools_.append(tools[val["idx"]])

        for n in response:
            tools_.append(tools[n.metadata["idx"]])

        tools_.append(tools[-1]) # add SQL tool
        #print("tools_: ", tools_)

        #return tools_
        return [adapt_to_async_tool(t) for t in tools_]

