# Knowledge base guide

1. Obtain a legally usable agricultural source and record its usage terms.
2. Save it under ignored `knowledge/source_docs/`.
3. Add one manifest row with a stable document ID and complete provenance.
4. Compute and add SHA-256 when the source is meant to be immutable.
5. Validate, ingest, and build the selected index using `docs/rag_guide.md`.
6. Inspect `GET /api/v1/knowledge/sources` and perform a citation spot check.

Do not crawl websites automatically. Do not add a URL without a local reviewed document. Do not
commit copyrighted source files, vector indexes, or extracted text unless redistribution is allowed.
The test fixture `SAMPLE_001` exists only for software tests and must never support agronomic claims.

In development, `POST /api/v1/rag/rebuild` rebuilds from the fixed project manifest and source folder.
The endpoint accepts no filesystem path and will not download an embedding model. It is disabled when
`APP_ENV` is not `development`.
