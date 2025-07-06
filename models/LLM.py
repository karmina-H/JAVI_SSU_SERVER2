"""
This module provides a class for interacting with a Language Model (LLM) using the ollama library.
"""

from typing import Dict, Iterator, List, Optional

from ollama import Client, ResponseError



class LLM():


    def __init__(self, **kwargs) -> None:

        self.messages: List[Dict[str, str]] = []

        self.system_prompt: Optional[str] = kwargs.get("system_prompt")

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})


        self.is_chat_history_disabled: Optional[bool] = kwargs.get("disable_chat_history")

        self.model = Client()

    def exists(self) -> bool:
        """
        Check if the specified LLM model exists.

        Returns:
            True if the model exists, False otherwise.
        """
        try:
            # Assert ollama model validity
            #_ = self.model.show(self.model_id)

            return True
        except ResponseError:
            return False

    def forward(self, message: str) -> str: # Change return type to str as it returns a full string
        message = message + "\n한국어 사용자 이므로 모든 질문과 답변은 한국어만 가능합니다."
        self.messages.append({"role": "user", "content": message})

        # Call the chat method without streaming (stream=False or omit it)
        response = self.model.chat(
            model='llama3.1:8b-instruct-q4_0',
            messages=self.messages,
            stream=False, # <--- Changed this to False or remove it if False is default
        )

        # Based on your previous 'chunk' structure, it might look like this:
        token = response["message"]["content"] # The entire generated content
        assistant_role = response["message"]["role"] # The role of the assistant

        generated_content = token # Since it's not streaming, 'token' is the full content
        
        if self.is_chat_history_disabled:
            self.messages.pop()
        else:
            # Append the full generated content to chat history
            self.messages.append({"role": assistant_role, "content": generated_content})
        
        return generated_content # Return the full generated text

#####################################


# """
# This module provides a class for interacting with a Language Model (LLM) using the transformers library.
# """

# from typing import Dict, Iterator, List, Optional

# import torch
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# from .common import BaseModel


# class LLM(BaseModel):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#         self.messages: List[Dict[str, str]] = []

#         self.system_prompt: Optional[str] = kwargs.get("system_prompt")
#         if self.system_prompt:
#             self.messages.append({"role": "system", "content": self.system_prompt})

#         self.is_chat_history_disabled: Optional[bool] = kwargs.get("disable_chat_history")

#         # Load model and tokenizer from Hugging Face
#         model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
#         self.pipeline = pipeline(
#             "text-generation",
#             model=model_id,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             device_map="auto",
#         )
#         self.pipeline.model.eval()
#         self.tokenizer = self.pipeline.tokenizer

#     def exists(self) -> bool:
#         """
#         Dummy check to confirm model is loaded.
#         """
#         return self.pipeline is not None

#     def forward(self, message: str) -> Iterator[str]:
#         """
#         Generate text from user input using the specified LLM.

#         Args:
#             message: The user input message.

#         Returns:
#             An iterator that yields the generated text in chunks.
#         """
#         self.messages.append({"role": "user", "content": message})

#         prompt = self.tokenizer.apply_chat_template(
#             self.messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         terminators = [
#             self.tokenizer.eos_token_id,
#             self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]

#         outputs = self.pipeline(
#             prompt,
#             max_new_tokens=2048,
#             eos_token_id=terminators,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9
#         )

#         full_output = outputs[0]["generated_text"]
#         generated_text = full_output[len(prompt):]

#         if self.is_chat_history_disabled:
#             self.messages.pop()
#         else:
#             self.messages.append({"role": "assistant", "content": generated_text})

#         # Yield output in chunks
#         for token in generated_text.split():
#             yield token + " "
