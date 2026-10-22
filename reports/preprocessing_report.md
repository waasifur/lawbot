# LawBot preprocessing report

## Input and row counts

| File | Input rows | Valid schema | Invalid schema |
|---|---:|---:|---:|
| `ipc_qa.json` | 2267 | 2267 | 0 |
| `crpc_qa.json` | 8194 | 8194 | 0 |
| `constitution_qa.json` | 4082 | 4082 | 0 |
| **Total** | **14543** | **14543** | **0** |

| Step | Rows remaining | Rows removed at step |
|---|---:|---:|
| Valid schema normalization | 14543 | 0 invalid records |
| Minimum lengths (10 characters each) | 13931 | 612 |
| Garbage/truncation filter | 13931 | 0 |
| Exact deduplication | 13910 | 21 |
| Near-duplicate question filter (MinHash > 0.95) | 13832 | 78 |
| Final train / validation | 11064 / 2768 | — |

## Source distribution

| Source | Total | Train | Validation |
|---|---:|---:|---:|
| IPC | 2171 | 1736 | 435 |
| CrPC | 7778 | 6222 | 1556 |
| Constitution | 3883 | 3106 | 777 |

## Length statistics (whitespace-separated words)

| Field | Count | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Question | 13832 | 17.5 | 16.0 | 3 | 136 |
| Answer | 13832 | 21.88 | 17.0 | 1 | 235 |

## Extracted references

3171 of 13832 records (22.9%) contain at least one extracted section/article reference.

## Before/after examples

Raw examples are shown alongside their cleaned, normalized records. The raw field mapping is `question`/`answer` to `instruction`/`output`.

### Example 1

- Before question: What is the title and extent of operation of the Indian Penal Code?
- Before answer: The title is 'The Indian Penal Code' and its operation extends to the punishment of offences committed within India, and beyond but which by law may be tried within India. It also includes extension of the Code to extra-territorial offences.
- After instruction: What is the title and extent of operation of the Indian Penal Code?
- After output: The title is 'The Indian Penal Code' and its operation extends to the punishment of offences committed within India, and beyond but which by law may be tried within India. It also includes extension of the Code to extra-territorial offences.
- References: []

### Example 2

- Before question: Where does the operation of 'The Indian Penal Code' extend to, and does it include extra-territorial offences?
- Before answer: The operation of 'The Indian Penal Code' extends to the punishment of offences committed within India, and beyond but which by law may be tried within India. It also includes extension of the Code to extra-territorial offences.
- After instruction: Where does the operation of 'The Indian Penal Code' extend to, and does it include extra-territorial offences?
- After output: The operation of 'The Indian Penal Code' extends to the punishment of offences committed within India, and beyond but which by law may be tried within India. It also includes extension of the Code to extra-territorial offences.
- References: []

### Example 3

- Before question: What does the Indian Penal Code stipulate about certain laws?
- Before answer: The Indian Penal Code specifies that certain laws are not to be affected by this Act.
- After instruction: What does the Indian Penal Code stipulate about certain laws?
- After output: The Indian Penal Code specifies that certain laws are not to be affected by this Act.
- References: []

## Notes and limitations

- Inputs contain `question` and `answer`; the processor maps these to `instruction` and `output`. The four optional CrPC `id` fields are not used.
- Source labels are inferred from filenames because the input rows do not provide a `source` key.
- Near-duplicate matching uses 16 seeded word-bigram MinHash permutations and removes questions with estimated similarity strictly above 0.95 (all signature positions must agree at this sketch size).
- Reference extraction retains matches from both questions and answers. Original answer text remains intact after text cleanup.
- The source data uses legacy IPC/CrPC terminology. The IPC, CrPC, and Evidence Act were replaced by the BNS, BNSS, and Bharatiya Sakshya Adhiniyam, effective 1 July 2024; this dataset does not map or verify those current-law equivalents.
- Source answers are retained as supplied and were not independently checked for legal accuracy or currency.
