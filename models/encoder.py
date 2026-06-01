from sentence_transformers import SentenceTransformer


class HypothesisEncoder:

    def __init__(

        self,

        model_name="sentence-transformers/all-MiniLM-L6-v2",

        device="cuda"

    ):

        self.model = SentenceTransformer(

            model_name,

            device=device

        )

        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):

        return self.model.encode(

            texts,

            convert_to_tensor=True

        )