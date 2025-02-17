import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";
import { promises as fs } from "fs";

export default async function (req, res) {
  const chat = new ChatOpenAI();

  const dataDirectory = path.join(process.cwd(), "data");
  const text = await fs.readFile(dataDirectory + "/blogathon.txt", "utf8");

  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  const chain = ConversationalRetrievalQAChain.fromLLM(
    chat,
    vectorStore.asRetriever(),
    {
      returnSourceDocuments: true,
    }
  );

  const response = await chain.call({
    question: req.body.question,
    chat_history: req.body.history,
  });

  res
    .status(200)
    .json({ result: response.text, references: response.sourceDocuments });
}
