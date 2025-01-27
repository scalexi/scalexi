import json
import os
import time
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from openai import OpenAI
import cohere
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from scalexi.utilities.logger import Logger
from scalexi.utilities.text_processing import remove_code_blocks
from scalexi.openai.pricing import OpenAIPricing
import pkgutil

# Create a logger file
logger = Logger().get_logger()

data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
pricing_info = json.loads(data)
#print(dfm.json_to_yaml(pricing_info))
pricing = OpenAIPricing(pricing_info)


class Generator:
    def __init__(self, openai_key=None, enable_timeouts= False, timeouts_options= None):
        self.openai_api_key = openai_key if openai_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key.")
        self.client = OpenAI(api_key=self.openai_api_key, max_retries=3)
        if enable_timeouts:
            if timeouts_options is None:
                timeouts_options = {"total": 120, "read": 60.0, "write": 60.0, "connect": 10.0}
                self.client = self.client.with_options(timeout=httpx.Timeout(120.0, read=60.0, write=60.0, connect=10.0))
            else:
                self.client = self.client.with_options(timeout=httpx.Timeout(timeouts_options["total"], timeouts_options["read"], timeouts_options["write"], timeouts_options["connect"]))
        

    def _remove_context_from_json(self, prompt_str):
        """
        Remove the 'request' field from a JSON string.

        This function takes a JSON string as input, converts it to a dictionary, 
        removes the 'request' field if present, and then converts it back to a JSON string 
        with formatting adjustments to remove braces and brackets.

        @param prompt_str: A JSON string from which the 'request' field needs to be removed.
        @return: A formatted JSON string without the 'request' field and without braces and brackets.

        Example:
        Input: '{"request": "data", "response": "data"}'
        Output: '"response": "data"'
        """
        prompt = json.loads(prompt_str)
        if "request" in prompt:
            del prompt["request"]
        logger.debug(prompt)
        result = json.dumps(prompt, indent=4).replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        return result

          
    def ask_gpt(self, request_params=None, temperature=0.7, 
            max_tokens=1000, top_p=1, frequency_penalty=0, 
            presence_penalty=0, stop=["END"], preprocess=True,
            request_type="DEFAULT", model_name="gpt-4o",
            system_prompt=None, user_prompt=None, stream=False):
        """
        Generate a response from GPT based on system and user prompts.

        This function sends a request to a GPT model and retrieves its completion. 
        It can automatically generate system and user prompts based on request parameters 
        and request type if they are not explicitly provided.

        @param request_params: Parameters for request customization (default: None).
        @param temperature: Sampling temperature (default: 0.7).
        @param max_tokens: Maximum number of tokens in the response (default: 1000).
        @param top_p: Nucleus sampling parameter (default: 1).
        @param frequency_penalty: Frequency penalty for token repetition (default: 0).
        @param presence_penalty: Presence penalty for token presence (default: 0).
        @param stop: Stop sequences for the completion (default: ["END"]).
        @param preprocess: Flag to indicate preprocessing (default: True).
        @param request_type: Type of request, affects prompt generation (default: "DEFAULT").
        @param model_name: Name of the GPT model to use (default: "gpt-4o").
        @param system_prompt: System prompt for the model (default: None).
        @param user_prompt: User prompt for the model (default: None).
        @return: A tuple containing the response text and a dummy value (0.0).
        @exception HTTPException: Raises an HTTPException with status code 500 on error.

        Usage example:
            response, _ = ask_gpt(request_params={'key': 'value'})
        """
        try:
            if system_prompt is None or user_prompt is None:
                logger.error(f'[SCALEXIGenerator] System or user prompt is None')
                raise Exception('[SCALEXI:ask_gpt] System or user prompt is None')

            logger.debug(system_prompt)
            logger.debug(user_prompt)
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                #stop=stop,
                stream=stream
            )
            
            end_time = time.time()  # End time
            execution_time = end_time - start_time  # Execution time
            token_usage = self.extract_gpt_token_usage(response)
            price = pricing.estimate_inference_cost(token_usage['prompt_tokens'], token_usage['completion_tokens'], model_name)
            #print(response)
            logger.debug(f'[SCALEXIGenerator] Response: {response}')
            logger.info(f'[SCALEXIGenerator] returning response, pricing, token_usage, execution_time')
            return response.choices[0].message.content, price, token_usage, execution_time
        except Exception as e:
            logger.error(f'[SCALEXIGenerator] Error in ask_gpt: {e}')
            raise HTTPException(status_code=500)
        
    def extract_gpt_token_usage(self, response):
        """
        Extracts the token usage from a ChatCompletion response object and returns it in a dictionary.

        :param response: The ChatCompletion response object.
        :return: A dictionary containing the number of tokens used for the prompt, completion, and total.
        """
        if hasattr(response, 'usage'):
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        else:
            token_usage = {"error": "No token usage information available"}

        return token_usage

            
    def ask_cohere(self, request_params=None, temperature=0.7, 
                       max_tokens=1000, 
                       top_p=1, 
                       frequency_penalty=0, 
                       presence_penalty=0, 
                       stop=["END"], 
                       preprocess=True,
                       request_type="DEFAULT",
                       model_name = "gpt-4o",
                       system_prompt=None,
                       user_prompt=None, 
                       is_chat_llm=True):
        print('ask_cohere started')
        start_time = time.time()  # Start time
        if not is_chat_llm:
            print('choere generative')
            co = cohere.Client('aLSGYTnNeq9JFXcyM4srY5knsKgizWlqKHEzmEMv') # This is your trial API key
            print (system_prompt+ ' '+ user_prompt)
            response = co.generate(
            model='command',
            prompt=system_prompt+ ' '+ user_prompt,
            max_tokens=4096,
            temperature=0.3,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')
            print('Prediction: {}'.format(response.generations[0].text))
            price = 0.0
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            end_time = time.time()  # End time
            execution_time = end_time - start_time  # Execution time
            return remove_code_blocks(response.generations[0].text), 0.0, price, token_usage, execution_time
            #return response.generations[0].text, 0.0
        else:
            try:
                print('choere chat')
                co = cohere.Client('aLSGYTnNeq9JFXcyM4srY5knsKgizWlqKHEzmEMv') # This is your trial API key
                response = co.chat( 
                    model='command',
                    message=system_prompt+ ' '+ user_prompt,
                    temperature=0.1,
                    chat_history=[],
                    prompt_truncation='AUTO',
                    stream=False,
                ) 
                print(remove_code_blocks(response.text))
                price = 0.0
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                end_time = time.time()  # End time
                execution_time = end_time - start_time  # Execution time
                return remove_code_blocks(response.text), 0.0, price, token_usage, execution_time
            except Exception as e:
                print('[ask_cohere] Exception: ', e)


    def ask_gemini_generative (self, request_params=None, temperature=0.7, 
                    max_output_tokens=1000, 
                    top_p=1, 
                    frequency_penalty=0, 
                    presence_penalty=0, 
                    max_retries=6,
                    n=1,
                    verbose=False,
                    preprocess=True,
                    request_type="DEFAULT",
                    model_name = "gemini-pro",
                    system_prompt=None,
                    user_prompt=None):  #gemini-pro or models/text-bison-001
        
        if (system_prompt is None) or (user_prompt is None):
            logger.error(f'[SCALEXIGenerator] System or user prompt is None')
        else:
            system_prompt = system_prompt
            user_prompt = user_prompt
            
        logger.debug(system_prompt)
        logger.debug(user_prompt)    
                    
        """
        At the command line, only need to run once to install the package via pip:

        $ pip install google-generativeai
        """

        #genai.configure(api_key="YOUR_API_KEY")

        # Set up the model
        generation_config = {
        "temperature": 0.3,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 4096,
        }

        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        ]
        start_time = time.time()  # Start time
        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)

        convo = model.start_chat(history=[
        ])
        convo.send_message(system_prompt+'\n'+user_prompt)
        print(convo.last.text)
        price = 0.0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        end_time = time.time()  # End time
        execution_time = end_time - start_time  # Execution time
        return convo.last.text, price, token_usage, execution_time


    def ask_gemini_chat (self, request_params=None, temperature=0.7, 
                    max_output_tokens=1000, 
                    top_p=1, 
                    frequency_penalty=0, 
                    presence_penalty=0, 
                    max_retries=6,
                    n=1,
                    verbose=False,
                    preprocess=True,
                    request_type="DEFAULT",
                    model_name = "gemini-pro",
                    system_prompt=None,
                    user_prompt=None):  #gemini-pro or models/text-bison-001
        
        if (system_prompt is None) or (user_prompt is None):
            logger.error(f'[SCALEXIGenerator] System or user prompt is None')
            raise Exception('System or user prompt is None')
        else:
            system_prompt = system_prompt
            user_prompt = user_prompt
            
        logger.debug(system_prompt)
        logger.debug(user_prompt)        
    
        llm = ChatGoogleGenerativeAI(model=model_name, max_output_tokens = max_output_tokens,
                                    top_p=1, 
                                    frequency_penalty=frequency_penalty, 
                                    presence_penalty=presence_penalty, 
                                    max_retries=max_retries,
                                    n=n,                                    
                                    verbose=verbose,
                                    google_api_key=os.getenv("GOOGLE_API_KEY"))
        
       
        prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                # New question
                ("user", user_prompt),
            ])        
        #print('prompt:',prompt)
        start_time = time.time()  # Start time
        response = llm.invoke(system_prompt+'\n'+user_prompt)# prompt | llm #llm.invoke(prompt)
        logger.debug(response)
        #exit(0)
        price = 0.0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        end_time = time.time()  # End time
        execution_time = end_time - start_time  # Execution time
        return response.content, price, token_usage, execution_time
       
    
    def ask_llm(self,request_params=None, 
                llm_type="gpt",  
                temperature=0.7, 
                max_output_tokens=1000, 
                max_tokens=1000, 
                top_p=1, 
                frequency_penalty=0, 
                presence_penalty=0, 
                stop=["END"], 
                preprocess=True,
                request_type="DEFAULT",
                model_name = "gpt-4o",
                system_prompt=None,
                user_prompt=None,
                stream=False):
            if "gpt" in model_name.lower():  # Using lower() to make the check case-insensitive
                logger.info(f"[Generator.ask_llm] The model name is a GPT model. {model_name}")
                return self.ask_gpt(request_params, 
                            temperature=temperature, 
                            max_tokens=max_tokens, 
                            top_p=top_p, 
                            frequency_penalty=frequency_penalty, 
                            presence_penalty=presence_penalty, 
                            stop=stop, 
                            preprocess=preprocess,
                            request_type=request_type,
                            model_name = model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            stream=stream)
            elif "command" in model_name.lower():  # Using lower() to make the check case-insensitive
                return self.ask_cohere(self,
                       temperature=temperature, 
                            system_prompt=system_prompt,
                            user_prompt=user_prompt, 
                            is_chat_llm=True)
            elif "gemini" in model_name.lower():  # Using lower() to make the check case-insensitive
                print("The model name is a Gemini model.", model_name)
                return self.ask_gemini_chat(request_params, temperature=temperature, 
                            max_output_tokens=max_output_tokens, 
                            top_p=top_p, 
                            frequency_penalty=frequency_penalty, 
                            presence_penalty=presence_penalty, 
                            preprocess=preprocess,
                            request_type=request_type,
                            model_name = model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt)
            else: 
                print('The model name not recognized.', model_name)
                return self.ask_gpt(request_params, temperature=temperature, 
                            max_tokens=max_tokens, 
                            top_p=top_p, 
                            frequency_penalty=frequency_penalty, 
                            presence_penalty=presence_penalty, 
                            stop=stop, 
                            preprocess=preprocess,
                            request_type=request_type,
                            model_name = model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            stream=stream)



