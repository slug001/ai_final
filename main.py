import os
import json
import pickle
from typing import List, Dict
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login



# LLM_MODEL = "microsoft/phi-2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
cache_dir = os.path.expanduser("~/data_18TB/")

DB_PATH = "./chroma_db"
KNOWLEDGEBASE_PATH = "Knowledge-Base.pkl" 
THRESHOLD = 0.80 

def get_vectordb() -> Chroma:
    embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    with open(KNOWLEDGEBASE_PATH, "rb") as f:
        docs: List[Document] = pickle.load(f)

    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=DB_PATH)
    vectordb.persist()
    return vectordb

# ========== Step 1: build LLM ==========
def get_llm(llm_model: str) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        device_map="auto",
        torch_dtype="auto",
        cache_dir=cache_dir,
    )

    tokenizer.pad_token = tokenizer.eos_token
    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

def RAG(query: str):
    # ========== Step 2: build knowledge ==========
    vectordb  = get_vectordb()
    retriever = vectordb.as_retriever()

    system_prompt = (
        """
            你是一位專家，請參考以下輸出格式，根據context回答input。
            
            目標：
                分析廣告文字內容，根據法律條款和案例判斷廣告用詞是否涉及誇大療效及違法，並提供違法機率評估。
                回應內容必須完全依照格式，且使用繁體中文。回應簡潔有力，不需要提供分析過程的文字。
            
            合規性判斷：
                - 無罪判定原則：不捏造或過度解讀廣告文字，從寬認定合法性，但如果是"藥物", "宣稱科學實證", "國外研究"一類用語，則提高違法可能性認定，除非內容中出現完整的『衛福部核可字號xxx』或『衛福部認證』。
                - 比對允許使用的廣告用詞："
                - 「完整補充營養」「調整體質」「促進新陳代謝」「幫助入睡」「保護消化道全機能」「改變細菌叢生態」「排便有感」「在嚴謹的營養均衡與熱量控制，以及適當的運動條件下，適量攝取本產品有助於不易形成體脂肪」等用語不視為違法風險。
                - 「能完整補充人體營養」「青春美麗」「排便順暢」…（略）
            
            分析步驟：
                1. 解析廣告內容
                2. 文件檢索與法規比對
                3. 判斷違法機率
                4. 裁罰依據：《食品安全衛生管理法》第45條（罰 4 萬至 400 萬元）
            
            回應格式：
                若違法機率 <= 80% \n
                違法機率: X % \n
                違法內容分析: 經分析廣告內容，違法風險較低。 \n
                罰款額度: 無。 \n
                參考依據: [vector store 文件 ID] \n

                若違法機率 > 80% \n
                違法機率: X % \n
                違法內容分析: 段落或句子: [違法描述], [裁罰原因] \n
                違反條款: [適用法規] \n
                罰款額度: [依據法規] \n
                參考依據: [vector store 文件 ID] \n
            
            回應結果範例:    
            {{
                "範例1": {{
                    "messages": [
                        {{
                            "role": "user",
                            "content": "請分析廣告內容：'本產品每日至少3~4次可隨餐食用，幫助維持消化道機能' 是否合法？"
                        }},
                        {{
                            "role": "assistant",
                            "content": "違法機率: 10%, 違法內容分析: 經分析廣告內容，違法風險較低。罰款額度: 無。參考依據: 食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-項次3"
                        }}
                    ]
                }},
                "範例2": {{
                    "messages": [
                        {{
                            "role": "user",
                            "content": "請分析廣告內容：'2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流' 是否合法？"
                        }},
                        {{
                            "role": "assistant",
                            "content": "違法機率: 90%, 違法內容分析: 段落或句子: '2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流', 由於承諾療效過度具體且容易誤導消費者對於醫療效益的期待。 違反條款: 違反《食品安全衛生管理法》第28條第1項。 罰款額度: 第45條第1項 違反規定者罰四至四百萬。 參考依據: 無。"
                        }}
                    ]
                }}
            }}

        請根據以下 context：
        {context}
        並回答以下問題：
        {input}
        """
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    llm = get_llm(LLM_MODEL)

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    rag_response = rag_chain.invoke({"input": query})

    answer_text = rag_response["answer"]
    print("\n=== RAG 回應 ===")
    print(answer_text)

    prob = 0.0
    for line in answer_text.splitlines():
        if line.strip().startswith("違法機率"):
            try:
                prob = float(line.split(":")[1].strip().rstrip("%")) / 100.0
            except Exception:
                pass
            break

    return 1 if prob > THRESHOLD else 0


if __name__ == "__main__":
    QUERY = (
        "【消化酵素是什麼？】 消化酵素是可將食物由大分子分解為小分子，幫助消化，維持消化道機能。 美國n.zimes®高活性綜合消化酵素 世界一流酵素大廠 n.zimes® ► 幫助消化效率大幅提升 ✦ 多元真菌發酵酵素 ► 比一般蔬果酵素效果更好 ✦ 綜合19種超級酵素配方 ► 活性酵素種類多 ✦ 荷蘭 TNO 實驗研究 ►實驗證實大幅提升消化效率 ✦ 保證絕不添加傷身瀉藥成分(番瀉葉、番瀉苷) ✦ 給你大自然的真實，無添加任何化學賦形劑、果汁粉、甜味劑 丹麥Gastro-AD®保衛樂 德式乳桿菌大豆發酵物 ✦ 世界大廠Gastro-AD® ► 緩解飲食後困擾 ✦ 不含西藥、制酸劑 ► 成分天然無副作用 ✦ 具GMP KOSHER HALA安全認證 【消化禮物 商品資訊】 產品名稱：[消化禮物]保衛高活性19種消化酵素 成分：大豆發酵物(非基改大豆、德氏乳桿菌質)、綜合消化酵素(麥芽糊精、澱粉分解酵素、植酸酵素、葡萄糖澱粉酵素RO、複合酵素(含脂肪分解酵素、蛋白質分解酵素、澱粉分解酵素、麥芽糊精)、中性蛋白質分解酵素、鳳梨酵素、蛋白質分解酵素4.5、轉化酵素、蛋白質分解酵素3.0、半纖維素分解酵素、果膠酵素、α-半乳糖苷酶、蛋白質分解酵素AM、脂肪分解酵素、胜肽分解酵素、乳糖分解酵素、蛋白質分解酵素6.0、澱粉酵素、纖維素分解酵素AN)、玉米來源可溶性纖維 膠囊成分：羥丙基甲基纖維素 內容量：560毫克/顆，30顆/盒 食用方式：每日2~4顆，隨餐食用。 保存期限：二年 保存方法：請置於陰涼乾燥處，避免日光直照或高溫潮濕的環境。開封後請將乾燥劑取出並於60天內食用完畢，以保最佳效果。 產地：台灣 過敏原：本產品含大豆製品及麩質之穀物製品。 全素可食"
    )
    flag = RAG(QUERY)
    print(f"\n合法(0) / 違法(1) 判定結果: {flag}")