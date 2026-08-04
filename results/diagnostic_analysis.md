# No-Context Diagnostic Analysis

## Purpose

The local Qwen2.5-3B-Instruct model was asked ten answerable factual questions without access to the Material UI corpus. The same questions were then answered using BM25 and dense retrieval with the same generation model and settings.

## Diagnostic composition

- 10 answerable questions
- 7 factoid questions
- 3 multi-hop questions

## Summary

| System | Fully correct | Rate |
|---|---:|---:|
| Naive, no context | 0/10 | 0% |
| BM25 RAG | 3/10 | 30% |
| Dense RAG | 8/10 | 80% |

The naive model produced 2 partially correct answers and 8 incorrect answers. Retrieval improved 9/10 answers.

## Per-question results

| Q | Type | Naive status | Naive score | BM25 score | Dense score |
|---:|---|---|---:|---:|---:|
| 1 | factoid | partially correct | 1/2 | 1/2 | 1/2 |
| 2 | factoid | incorrect | 0/2 | 0/2 | 2/2 |
| 3 | factoid | incorrect | 0/2 | 2/2 | 2/2 |
| 4 | factoid | incorrect | 0/2 | 2/2 | 2/2 |
| 5 | factoid | incorrect | 0/2 | 2/2 | 2/2 |
| 6 | factoid | partially correct | 1/2 | 0/2 | 2/2 |
| 7 | factoid | incorrect | 0/2 | 0/2 | 2/2 |
| 8 | multi-hop | incorrect | 0/2 | 0/2 | 2/2 |
| 9 | multi-hop | incorrect | 0/2 | 1/2 | 2/2 |
| 10 | multi-hop | incorrect | 0/2 | 1/2 | 0/2 |

## Representative diagnostic failures

### Question 1

**Question:** Does Material UI include a Menubar component out of the box, and what do I need to install to use the menubar shown in the docs?

**Reference:** No. Material UI does not include a Menubar component out of the box. The Menubar shown in the documentation is composed using Base UI, so you must install the @base-ui/react package.

**Naive answer:** Material-UI does not include a built-in Menubar component out of the box. To use a Menubar from the documentation, you would typically need to install the `@mui/material` package or import components directly from MUI. However, for a Menubar, you might need to look into third-party packages or create your own custom solution using MUI's other components.

**Dense RAG answer:** Material UI does not include a menu bar component out of the box [1]. To use the menubar shown in the docs, you need to install Base UI first [1], which is required because the menubar components are composed with Base UI's `Menubar` component and styled to align with Material Design specifications [1]. After installing Base UI, you can copy and paste the provided code snippets into your project [1].

### Question 2

**Question:** Which prop keeps the CircularProgress track visible at all times?

**Reference:** Pass the enableTrackSlot prop to keep the CircularProgress track visible at all times.

**Naive answer:** The prop that keeps the CircularProgress track visible at all times is `trackProps`.

**Dense RAG answer:** Prop `enableTrackSlot`: type bool. Default: false. [2]

### Question 7

**Question:** Which prop limits how many avatars an AvatarGroup shows, and what is its default value?

**Reference:** The max prop limits how many avatars an AvatarGroup shows, and its default value is 5.

**Naive answer:** The prop that limits how many avatars an AvatarGroup shows is `count`, and its default value is 3.

**Dense RAG answer:** Prop `max` limits how many avatars an AvatarGroup shows, and its default value is 5. [1] [2]

### Question 8

**Question:** Which Rating prop sets the minimum increment the value can change by, and what is that prop's default?

**Reference:** The precision prop sets the minimum increment by which the Rating value can change, and its default value is 1.

**Naive answer:** The prop that sets the minimum increment the value can change by is `step`, and its default value is 0.1.

**Dense RAG answer:** Prop `precision` sets the minimum increment the value can change by [2][3]. The default is 1 [2][3].

## Conclusion

The local LLM answered none of the ten diagnostic questions fully correctly without retrieved context. Retrieval improved 9 of the ten answers, demonstrating that the selected corpus contains information the model did not reliably know from its parameters alone.

The diagnostic therefore satisfies the corpus-selection check: the model could sometimes produce related or partially correct information, but it did not fully answer any of the ten questions without retrieval. Dense retrieval produced the strongest overall improvement.
