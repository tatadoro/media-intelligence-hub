/* sql/06_update_view_articles_dedup_score_prefers_nlp.sql */

DROP VIEW IF EXISTS media_intel.articles_dedup;

CREATE VIEW media_intel.articles_dedup AS
SELECT
    dedup_key,
    argMax(id, score) AS id,
    argMax(title, score) AS title,
    argMax(link, score) AS link,
    argMax(published_at, score) AS published_at,
    argMax(source, score) AS source,
    argMax(raw_text, score) AS raw_text,
    argMax(clean_text, score) AS clean_text,
    argMax(nlp_text, score) AS nlp_text,
    argMax(summary, score) AS summary,
    argMax(keywords, score) AS keywords,
    argMax(lang, score) AS lang,
    argMax(keyphrases, score) AS keyphrases,
    argMax(sentiment_label, score) AS sentiment_label,
    argMax(sentiment_score, score) AS sentiment_score,
    argMax(text_length_chars, score) AS text_length_chars,
    argMax(num_sentences, score) AS num_sentences,
    argMax(num_keywords, score) AS num_keywords,
    argMax(batch_id, score) AS batch_id,
    argMax(is_digest, score) AS is_digest,
    argMax(persons_actions, score) AS persons_actions,
    argMax(actions_verbs, score) AS actions_verbs,
    argMax(entities_persons, score) AS entities_persons,
    argMax(entities_persons_canon, score) AS entities_persons_canon,
    argMax(entities_orgs, score) AS entities_orgs,
    argMax(entities_geo, score) AS entities_geo,
    argMax(num_persons, score) AS num_persons,
    argMax(num_orgs, score) AS num_orgs,
    argMax(num_geo, score) AS num_geo,
    argMax(ingest_object_name, score) AS ingest_object_name,
    argMax(clean_len, score) AS best_clean_len
FROM
(
    SELECT
        a.*,

        /* ---- dedup_key ----
           Стабильный ключ дедупликации для всех источников:
           1) link, если есть
           2) иначе id
           Это предотвращает ситуацию, когда один и тот же id “разъезжается”
           по разным dedup_key из-за различий текста/хэша между загрузками.
        */
        coalesce(nullIf(a.link, ''), nullIf(a.id, ''), '') AS dedup_key,

        lengthUTF8(coalesce(a.clean_text, '')) AS clean_len,

        /* “есть тело” — синхронизируем с pipeline (min_len=400) */
        toUInt8(clean_len >= 400) AS has_body,

        /* есть ли “что-то” NLP, кроме голого текста */
        toUInt8(
            (a.lang != 'unknown')
            OR notEmpty(coalesce(a.keyphrases, ''))
            OR (a.sentiment_label != 'neu')
            OR (coalesce(a.sentiment_score, 0) != 0)
        ) AS has_nlp_extras,

        /* важно: канон персон */
        toUInt8(notEmpty(coalesce(a.entities_persons_canon, ''))) AS has_persons_canon,

        /* важно: sentiment */
        toUInt8(
            (a.sentiment_label != 'neu')
            OR (coalesce(a.sentiment_score, 0) != 0)
        ) AS has_sentiment,

        /* важно: actions */
        toUInt8(
            notEmpty(coalesce(a.persons_actions, ''))
            OR notEmpty(coalesce(a.actions_verbs, ''))
        ) AS has_actions,

        /* тай-брейкер по “времени загрузки” из batch_id */
        if(
            startsWith(a.batch_id, '20'),
            parseDateTimeBestEffortOrZero(a.batch_id),
            parseDateTimeBestEffortOrZero(extract(a.batch_id, '(\\d{8}T\\d{6}Z)'))
        ) AS loaded_at,

        /* score: чем левее — тем важнее */
        (
            has_body,
            clean_len,
            has_nlp_extras,
            has_persons_canon,
            has_sentiment,
            has_actions,
            loaded_at,
            a.published_at
        ) AS score
    FROM media_intel.articles AS a
) AS t
GROUP BY dedup_key
;