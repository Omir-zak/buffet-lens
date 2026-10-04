-- Run by Postgres once, when the volume is first created. A separate database for Langfuse
-- keeps its tables apart from the app's.
CREATE DATABASE langfuse;
