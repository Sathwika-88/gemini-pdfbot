from pydantic import BaseModel
import google.generativeai as genai
import instructor
from pydantic import BaseModel
from typing import List
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric
)
from deepeval import evaluate

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"
    
class ResponseSchema(BaseModel):
    answer: str

custom_llm = CustomGeminiFlash()

contextual_precision = ContextualPrecisionMetric(model=custom_llm)
contextual_recall = ContextualRecallMetric(model=custom_llm)
contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)
answer_relevancy = AnswerRelevancyMetric(model = custom_llm)
faithfulness = FaithfulnessMetric(model = custom_llm)

tc1 = LLMTestCase(input = " explain Types of Operating Systems", 
                  actual_output = """Batch OS–A set of similar jobs are stored in the main memory for execution. A job gets assigned to the CPU, only when the execution of the previous job completes.
                  2. Multiprogramming OS– The main memory consists of jobs waiting for CPU time. TheOS selects one of the processes and assigns it to the CPU. Whenever the executingprocess needs to wait for any other operation (like I/O), the OS selects another processfrom the job queue and assigns it to the CPU. This way, the CPU is never kept idleand the user gets the flavor of getting multiple tasks done at once.
                  3. Multitasking OS– Multitasking OS combines the benefits of Multiprogramming OSand CPU scheduling to perform quick switches between jobs. The switch is so quickthat the user can interact with each program as it runs.
                  4. Time Sharing OS– Time-sharing systems require interaction with the user to instruct
 the OS to perform various tasks. The OS responds with an output. The instructions are
 usually given through an input device like the keyboard.
 5. Real Time OS– Real-Time OS are usually built for dedicated systems to accomplish a
 specific set of tasks within deadlines.""",
                  expected_output = """ 1. Batch OS – A set of similar jobs are stored in the main memory for execution. A job gets assigned to the CPU, only when the execution of the previous job completes.
                  Multiprogramming OS – The main memory consists of jobs waiting for CPU time. The OS selects one of the processes and assigns it to the CPU. Whenever the executing process needs to wait for any other operation (like I/O), the OS selects another process from the job queue and assigns it to the CPU. This way, the CPU is never kept idle and the user gets the flavor of getting multiple tasks done at once.Multitasking OS
                    – Multitasking OS combines the benefits of Multiprogramming OS and CPU scheduling to perform quick switches between jobs. The switch is so quick that the user can interact with each program as it runs.Time Sharing 
                    OS – Time-sharing systems require interaction with the user to instruct the OS to perform various tasks. The OS responds with an output. The instructions are usually given through an input device like the keyboard.Real Time OS
                      – Real-Time OS are usually built for dedicated systems to accomplish a specific set of tasks within deadlines.""",
                  retrieval_context = [""" AnOperating System can be defined as an interface between user and hardware. It
 is responsible for the execution of all the processes, Resource Allocation, CPU
 management, File Management and many other tasks. The purpose of an operating
 system is to provide an environment in which a user can execute programs in a
 convenient and efficient manner.
 ● Types of Operating Systems :
 1. Batch OS–A set of similar jobs are stored in the main memory for execution. A job gets
 assigned to the CPU, only when the execution of the previous job completes.
 2. Multiprogramming OS– The main memory consists of jobs waiting for CPU time. The
 OS selects one of the processes and assigns it to the CPU. Whenever the executing
 process needs to wait for any other operation (like I/O), the OS selects another process
 from the job queue and assigns it to the CPU. This way, the CPU is never kept idle
 and the user gets the flavor of getting multiple tasks done at once.
 3. Multitasking OS– Multitasking OS combines the benefits of Multiprogramming OS
 and CPU scheduling to perform quick switches between jobs. The switch is so quick
 that the user can interact with each program as it runs.
 4. Time Sharing OS– Time-sharing systems require interaction with the user to instruct
 the OS to perform various tasks. The OS responds with an output. The instructions are
 usually given through an input device like the keyboard.
 5. Real Time OS– Real-Time OS are usually built for dedicated systems to accomplish a
 specific set of tasks within deadlines
."""])




#TC - 3
contextual_precision.measure(tc1)
print("Contextual Precision Score: ", contextual_precision.score)
print("Reason: ", contextual_precision.reason)

contextual_recall.measure(tc1)
print("Contextual Recall Score: ", contextual_recall.score)
print("Reason: ", contextual_recall.reason)

contextual_relevancy.measure(tc1)
print("Contextual Relevancy Score: ", contextual_relevancy.score)
print("Reason: ", contextual_relevancy.reason)

answer_relevancy.measure(tc1)
print("Answer Relevancy Score: ", answer_relevancy.score)
print("Reason: ", answer_relevancy.reason)

faithfulness.measure(tc1)
print("Faithfulness Score: ", faithfulness.score)
print("Reason: ", faithfulness.reason)