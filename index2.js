import axios from 'axios';
import { OpenAIApi, Configuration } from 'openai';
import similarity from 'compute-cosine-similarity';
import * as dotenv from 'dotenv';
import fs from 'fs/promises';

dotenv.config();
export const COMPLETIONS_MODEL = 'text-davinci-003';
export const DOC_EMBEDDINGS_MODEL = 'text-embedding-ada-002';

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

const getEmbedding = async (text) => {
    try {
        const { data: result } = await axios.post(
            'https://api.openai.com/v1/embeddings',
            {
                input: text,
                model: DOC_EMBEDDINGS_MODEL,
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
                },
            },
        );
        // console.log(result?.data[0]?.embedding);
        return result?.data[0]?.embedding;
    } catch (error) {
        console.error(error);
    }
};

async function documentoMaisSimilar(pergunta) {
    try {
        const arquivoRespostasEmbeddings = await fs.readFile('data/testeResumeEmbeddings.json', { encoding: 'utf8' });
        const arquivoRespostas = await fs.readFile('data/testeResume.json', { encoding: 'utf8' });

        const objRespostas = JSON.parse(arquivoRespostas);
        const objRespostasEmbeddings = JSON.parse(arquivoRespostasEmbeddings);
        const objPerguntasEmbeddings = await getEmbedding(pergunta);

        const similaridades = [];

        objRespostasEmbeddings.forEach((item, index) => {
            const embeddingsResposta = Object.entries(item)
                .filter((c) => c[0] !== 'title' && c[0] !== 'heading')
                .map((i) => i[1]);
            similaridades.push({ id: index, similaridade: similarity(embeddingsResposta, objPerguntasEmbeddings) });
        });

        const conteudosOrdenadosPorSimilaridade = similaridades.sort((a, b) => a.similaridade - b.similaridade).reverse();

        const melhorConteudoParaResposta = objRespostas[conteudosOrdenadosPorSimilaridade[0].id];
        return melhorConteudoParaResposta;
    } catch (error) {
        console.error(error);
    }
}

async function fazPergunta(prompt) {
    try {
        const completion = await openai.createCompletion({
            temperature: 0,
            max_tokens: 300,
            top_p: 1,
            frequency_penalty: 0,
            presence_penalty: 0,
            model: COMPLETIONS_MODEL,
            prompt: prompt,
        });

        return completion.data.choices[0].text;
    } catch (error) {
        console.error(error);
    }
}

async function main() {
    const pergunta = 'How did the pandemic affect the Olympic qualifiers?';
    const documento = await documentoMaisSimilar(pergunta);

    const prompt = `Reponda minha pergunta em portugues e de forma sucinta!
                    Answer the question as truthfully as possible using the provided context, 
                    and if the answer is not contained within the text below, say "I don't know." 

                    Context:
                       ### ${documento.title} ${documento.content}
                    
                    Pergunta: ${pergunta}

                    Resposta:
    `;

    const resposta = await fazPergunta(prompt);
    console.log(resposta);
}

main();
