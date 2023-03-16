import { OpenAIApi, Configuration } from 'openai';
import * as dotenv from 'dotenv';
import { parse } from 'csv-parse';
import fs from 'fs';

dotenv.config();

const COMPLETIONS_MODEL = 'text-davinci-003';
const MODEL_NAME = 'curie';
const DOC_EMBEDDINGS_MODEL = `text-search-${MODEL_NAME}-doc-001`;
const QUERY_EMBEDDINGS_MODEL = `text-search-${MODEL_NAME}-query-001`;

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

try {
    const completion = await openai.createCompletion({
        temperature: 0,
        max_tokens: 300,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
        model: COMPLETIONS_MODEL,
        prompt: 'Quem ganhou os jogos olimpicos de 2016?',
    });

    console.log(completion.data.choices[0].text);
} catch (error) {
    if (error.response) {
        console.log(error.response.status);
        console.log(error.response.data);
    } else {
        console.log(error.message);
    }
}

function getEmbedding(input, model) {
    const embedd = openai.createEmbeddingRequest();
    const result = openai.createEmbedding({ model, input });
    return result['data'][0]['embedding'];
}
const getDocEmbedding = (text) => getEmbedding(text, DOC_EMBEDDINGS_MODEL);
const getQueryEmbedding = (text) => getEmbedding(text, QUERY_EMBEDDINGS_MODEL);

function compute_doc_embeddings(dataFrame) {
    const embeddings = {};

    dataFrame.forEach((row, idx) => {
        const content = row.content.replace(/\n/g, ' ');
        const embedding = getDocEmbedding(content);
        embeddings[idx] = embedding;
        //const key = [idx, row.author];
    });

    return embeddings;
}

// function loadEmbeddingsJson(file){
//     const maxColunas = Object.entries(file[0]).map(i => i[0]).filter((c) => c !== "title" && c !== "heading");

// }

function loadEmbeddings(fname) {
    fs.readFile(fname, function (err, fileData) {
        if (err) return console.error(err);

        parse(fileData, { columns: true, skip_empty_lines: true }, function (err, rows) {
            // Object.entries(list[0]).map(i => i[0]).filter((c) => c !== "title" && c !== "heading")
            const max_dim = Math.max(...rows[0].filter((c) => c !== 'title' && c !== 'heading').map((c) => parseInt(c)));
            const data = {};
            rows.forEach((r) => {
                const key = [r.title, r.heading];
                const values = [];
                for (let i = 0; i <= max_dim; i++) {
                    values.push(r[i.toString()]);
                }
                data[key] = values;
            });
            return data;
        });
    });
}

function vectorSimilarity(x, y) {
    // Poderíamos usar similaridade de cosseno ou produto escalar para calcular a similaridade entre vetores.
    // Na prática, descobrimos que faz pouca diferença.

    const dotProduct = x.reduce((acc, val, i) => acc + val * y[i], 0);
    return dotProduct;
}

function orderDocumentSectionsByQuerySimilarity(query, contexts) {
    const query_embedding = getQueryEmbedding(query);
    const document_similarities = [];

    for (let doc_index = 0; doc_index < contexts.length; doc_index++) {
        const doc_embedding = contexts[doc_index];
        document_similarities.push(vectorSimilarity(query_embedding, doc_embedding));
    }

    return document_similarities.sort().reverse();
}
