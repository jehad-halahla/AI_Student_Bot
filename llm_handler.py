from llm import GeminiLLM
from chroma_text_processing import ChromaInterface
from chroma_text_processing import TextSplitter, RecursiveCharacterTextSplitterAdapter, NLTKTextSplitterAdapter, CustomSentenceTransformerEmbedding, ChromaInterface


class LLMHandler:
    PROMPT_TEMPLATES = {
        # Arabic Templates
        'default_ar': """
        معلومات السياق أدناه.
        ---------------------
        {context}
        ---------------------
        بناءً على معلومات السياق ، أجب على الاستفسار.
        الاستفسار: {query}
        اريد اجابة مفصلة ودقيقة: """,
        
        'detailed_ar': """
        المعلومات المتوفرة:
        ---------------------
        {context}
        ---------------------
        بناءً على المعلومات المتوفرة فقط، يرجى تقديم إجابة مفصلة وشاملة.
        الاستفسار: {query}
        الإجابة: """,

        'brief_ar': """
        السياق:
        ---------------------
        {context}
        ---------------------
        استنادًا إلى السياق أعلاه، قدم إجابة مختصرة على الاستفسار.
        الاستفسار: {query}
        """,

        # English Templates
        'default_en': """
        Context information below:
        ---------------------
        {context}
        ---------------------
        Based on the context information, answer the inquiry.
        Inquiry: {query}
        Please provide a detailed and accurate answer: """,
        
        'detailed_en': """
        Available information:
        ---------------------
        {context}
        ---------------------
        Based on the available information, please provide a detailed and comprehensive answer.
        Inquiry: {query}
        Answer: """,

        'brief_en': """
        Context:
        ---------------------
        {context}
        ---------------------
        Based on the context above, provide a brief answer to the inquiry.
        Inquiry: {query}
        """,
        
        # New robust template in English
        'robust_en': """
        Context information (if available):
        ---------------------
        {context}
        ---------------------
        Based on the context information provided (if any) and your general knowledge, please answer the following inquiry. If the context doesn't contain sufficient information, use your best judgment to provide a helpful response.

        Inquiry: {query}

        Instructions:
        1. If the context is relevant, use it to formulate your answer.
        2. If the context is insufficient or irrelevant, rely on your general knowledge to provide the best possible answer.
        3. If you're unsure about specific details, acknowledge this in your response.
        4. Provide a clear and concise answer, offering to elaborate if needed.

        Please proceed with your response:
        """,
        
        # New robust template in Arabic
        'robust_ar': """
        معلومات السياق (إن وجدت):
        ---------------------
        {context}
        ---------------------
        بناءً على معلومات السياق المقدمة (إن وجدت) ومعرفتك العامة، يرجى الإجابة على الاستفسار التالي. إذا لم يحتوِ السياق على معلومات كافية، استخدم أفضل تقديراتك لتقديم إجابة مفيدة.

        الاستفسار: {query}

        التعليمات:
        1. إذا كان السياق ذا صلة، استخدمه لصياغة إجابتك.
        2. إذا كان السياق غير كافٍ أو غير ذي صلة، اعتمد على معرفتك العامة لتقديم أفضل إجابة ممكنة.
        3. إذا كنت غير متأكد من تفاصيل معينة، اعترف بذلك في إجابتك.
        4. قدم إجابة واضحة وموجزة، مع عرض التوسع إذا لزم الأمر.

        يرجى المتابعة بإجابتك:
        """,



        # New context-restricted template in English
        'context_restricted_en': """
        Context information below:
        ---------------------
        {context}
        ---------------------
        Based on the context information provided, please answer the inquiry **only** if the context is directly relevant. If the context does not contain relevant information, simply state that no relevant information is available.

        Inquiry: {query}

        Instructions:
        1. Ensure that the answer is strictly based on the context information provided.
        2. If there is no direct connection between the context and the query, do not attempt to answer the query. Instead, respond with: "The context does not contain relevant information to answer the inquiry."
        3. If there is a connection, provide a precise and context-specific answer.

        Please proceed with your response:
        """,
        
        # New context-restricted template in Arabic
        'context_restricted_ar': """
        معلومات السياق أدناه:
        ---------------------
        {context}
        ---------------------
        بناءً على المعلومات الواردة في السياق، يرجى الإجابة على الاستفسار **فقط** إذا كانت المعلومات ذات صلة مباشرة. إذا لم يحتوي السياق على معلومات ذات صلة، يرجى ببساطة الإشارة إلى أن المعلومات غير متوفرة.

        الاستفسار: {query}

        التعليمات:
        1. تأكد من أن الإجابة تعتمد بشكل صارم على المعلومات الواردة في السياق.
        2. إذا لم يكن هناك صلة مباشرة بين السياق والاستفسار، لا تحاول الإجابة على الاستفسار. بدلاً من ذلك، استجب بـ: "لا يحتوي السياق على معلومات ذات صلة للإجابة على الاستفسار."
        3. إذا كانت هناك صلة، قدم إجابة دقيقة ومبنية على السياق.

        يرجى المتابعة بإجابتك:
        """
    }


    def __init__(self, collection_name, db_path, llm, template_name='default_en'):
        self.llm = llm
        # Initialize the interface
        text_splitter = RecursiveCharacterTextSplitterAdapter(chunk_size=200, chunk_overlap=20)
        self.chroma_interface = ChromaInterface(collection_name, db_path, text_splitter=text_splitter)
        self.template = self.PROMPT_TEMPLATES.get(template_name, self.PROMPT_TEMPLATES['default_en'])

    def generate_response(self, query):
        # Retrieve relevant information from Chroma
        query_results = self.chroma_interface.query(query)
        print(f"query_results == {query_results}")
        # Construct the prompt
        prompt = self._construct_prompt(query, query_results)

        # Generate response using the LLM
        response = self.llm.generate_content(prompt)

        return response

    def _construct_prompt(self, query, context):
        # Merge context information into a single string, with each item on a new line
        context_str = "\n".join(f"- {item}" for item in context)

        # Fill the selected template
        prompt = self.template.format(context=context_str, query=query)
        
        return prompt