import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";
import { promises as fs } from "fs";

export default async function (req, res) {
  const model = new OpenAI({});

  const dataDirectory = path.join(process.cwd(), "data");
  const text = await fs.readFile(dataDirectory + "/sparkloop.txt", "utf8");

  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever()
  );

  const response = await chain.call({
    question: req.body.question,
    chat_history: req.body.history,
  });

  console.log(response);

  res.status(200).json({ result: response.text });
}
