import sqlite3

from app.database.database import get_connection
from app.models.chunk_model import Chunk


class DocumentRepository:

    def list_documents(self, limit=100, offset=0):
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT d.id, d.source_type, d.source_name, d.author, d.created_at,
                       d.total_pages, COUNT(c.chunk_id) AS chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                LIMIT ? OFFSET ?
                """, (limit, offset)
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_documents(self):
        conn = get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        finally:
            conn.close()

    def get_document(self, document_id):
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT d.id, d.source_type, d.source_name, d.author, d.created_at,
                       d.total_pages, COUNT(c.chunk_id) AS chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = ? GROUP BY d.id
                """, (document_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["records"] = [dict(item) for item in conn.execute(
                "SELECT id, location, text FROM pages WHERE document_id = ? ORDER BY id",
                (document_id,),
            ).fetchall()]
            return result
        finally:
            conn.close()

    def browse_chunks(self, query=None, source_type=None, document_id=None, limit=20, offset=0):
        clauses = []
        params = []
        if query:
            clauses.append("(text LIKE ? OR source_name LIKE ? OR author LIKE ?)")
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern])
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        if document_id:
            clauses.append("document_id = ?")
            params.append(document_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute("SELECT COUNT(*) FROM chunks" + where, params).fetchone()[0]
            rows = conn.execute(
                "SELECT chunk_id, document_id, source_type, source_name, author, created_at, location, chunk_index, text, start_char, end_char FROM chunks"
                + where + " ORDER BY created_at DESC, document_id, chunk_index LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return total, [dict(row) for row in rows]
        finally:
            conn.close()

    def counts(self):
        conn = get_connection()
        try:
            return {
                "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "records": conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
            }
        finally:
            conn.close()

    def get_all_chunks(self):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT chunk_id, document_id, source_type, source_name, author,
                   created_at, location, chunk_index, text, start_char, end_char
            FROM chunks
            ORDER BY created_at, document_id, chunk_index, chunk_id
            """
        )

        chunks = [Chunk(*row) for row in cur.fetchall()]
        conn.close()

        return chunks

    def find_document_by_hash(self, file_hash):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, source_name
            FROM documents
            WHERE file_hash = ?
            LIMIT 1
            """,
            (file_hash,)
        )

        row = cur.fetchone()
        conn.close()

        return row

    def delete_document(self, document_id):
        conn = get_connection()
        cur = conn.cursor()

        # Delete chunks belonging to this document
        cur.execute(
            """
            DELETE FROM chunks
            WHERE document_id = ?
            """,
            (document_id,)
        )

        # Delete extracted pages
        cur.execute(
            """
            DELETE FROM pages
            WHERE document_id = ?
            """,
            (document_id,)
        )

        # Delete document record
        cur.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (document_id,)
        )

        conn.commit()
        conn.close()

    def save_document(self, document, file_hash):

        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO documents(
                id,
                source_type,
                source_name,
                author,
                created_at,
                total_pages,
                file_hash
            )
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                document.id,
                document.source_type,
                document.source_name,
                document.author,
                str(document.created_at),
                len(document.pages),
                file_hash
            )
        )

        conn.commit()
        conn.close()

    def save_pages(self, document):

        conn = get_connection()
        cur = conn.cursor()

        for page in document.pages:

            cur.execute(
                """
                INSERT INTO pages(
                    document_id,
                    location,
                    text
                )
                VALUES (?,?,?)
                """,
                (
                    document.id,
                    page.location,
                    page.text
                )
            )

        conn.commit()
        conn.close()

    def save_chunks(self, chunks):

        conn = get_connection()
        cur = conn.cursor()

        for chunk in chunks:

            cur.execute(
                """
                INSERT INTO chunks
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source_type,
                    chunk.source_name,
                    chunk.author,
                    chunk.created_at,
                    chunk.location,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char
                )
            )

        conn.commit()
        conn.close()
