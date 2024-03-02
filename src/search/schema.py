from pymilvus import FieldSchema, CollectionSchema, DataType

def token_schema(dimensions: int) -> CollectionSchema:
	fields = [
		FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
		FieldSchema(name="token", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
	]
	schema = CollectionSchema(fields, "Saves the token encoding of a position together with the position embedding.")
	return schema

def embedding_schema(dimensions: int) -> CollectionSchema:
	fields = [
		FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
		FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions)
	]
	schema = CollectionSchema(fields, "Saves the token encoding of a position together with the position embedding.")
	return schema

def get_index_definition(definition: str) -> dict:
	perdefined_indices = {
		"flat": {
			"index_type": "IVF_FLAT",
			"metric_type": "L2",
			"params": {"nlist": 128},
		}
	}
	return perdefined_indices[definition]
