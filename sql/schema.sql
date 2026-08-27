-- Reports, their pages and the passages that are searched.
-- A passage never crosses a page, so every hit can be cited by page.

create extension if not exists vector;

create table if not exists report (
    report_id    text primary key,
    company      text not null,
    fiscal_year  integer not null,
    title        text not null,
    source_url   text not null,
    sha256       text not null,
    pages        integer not null,
    indexed_at   timestamptz not null default now()
);

create table if not exists page (
    report_id  text not null references report on delete cascade,
    page_no    integer not null,          -- position in the PDF, starting at 1
    text       text not null,
    primary key (report_id, page_no)
);

create table if not exists passage (
    passage_id  bigserial primary key,
    report_id   text not null,
    page_no     integer not null,
    position    integer not null,         -- order of the passage on its page
    text        text not null,
    embedding   vector(384) not null,
    unique (report_id, page_no, position),
    foreign key (report_id, page_no) references page on delete cascade
);

create index if not exists passage_embedding_idx on passage using hnsw (embedding vector_cosine_ops);
