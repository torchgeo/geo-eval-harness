---
name: code-review
description: Review geo-eval-harness pull requests for correctness, security, unnecessary code and tests, duplicated behavior, and inflated prose. Use during automated Copilot PR reviews and requested code reviews. Keep findings brief and conversational.
---

# Code review

Review the changed code and its callers before suggesting a fix. Read the repository's `AGENTS.md` and relevant configuration. Apply the stricter preferences below to newly added code even where older examples allow explanatory comments or private helpers. Review the implementation, not whether its author used an LLM.

## Tests that earn their place

- A test should catch a plausible regression or verify a supported behavior. Identify the failure it would detect. Flag tests that merely repeat an assignment, check that a mock returns its configured value, compare a result with itself, or recompute the expected value with the same function being tested.
- Test observable behavior through the existing interface. Flag source-text searches, assertions about helper names or call order, and mocks that replace the behavior under test. Keep checks on ordering or calls when those are themselves required behavior, such as preprocessing exactly once or avoiding weight downloads during catalog discovery.
- Extend an existing test or parameterization when it covers the same behavior. Flag duplicated fixtures, repeated cases with no distinct failure mode, and large mock setups where a small real input would exercise the code.
- Do not request tests just because a file changed or coverage could increase. Documentation and instruction edits do not need tests that lock in their wording. Preserve useful regression tests and boundary cases for supported inputs, error handling, split isolation, band order, normalization, and result/resume identity.

## No explanatory code comments

- Flag added inline or block comments, including section banners, numbered steps, commented-out code, and narration of the next statement. Ask for clearer names or simpler code; put necessary design rationale in documentation or the PR description. Do not propose adding comments as the fix.
- Keep required license notices and functional directives such as shebangs, justified lint/type suppressions, and the repository's mandatory `# allow-except: <reason>` annotations. Concise Google-style API docstrings remain required where Ruff enforces them; remove filler and implementation walkthroughs from docstrings.

## Reuse and straightforward code

- Search the repository for equivalent behavior before judging a new function, including helpers with different names and inline implementations. When behavior already exists, cite its file and symbol and suggest reusing or narrowly extending it, updating callers, and removing the superseded copy. Do not combine distinct task or backend semantics just because their signatures look alike.
- Flag new project-owned private functions or methods with leading underscores. Prefer extending the existing function, inlining a trivial single-use helper, or a plainly named function with a clear responsibility. Preserve required interface hooks such as `BenchModel._forward_patch_features`, external overrides, and Python special methods. Do not request renaming untouched internals or breaking an existing interface to satisfy a naming preference.
- Flag one-line forwarding wrappers, classes that only wrap a function, factories for a single implementation, speculative plugin systems, and new configuration switches without a current caller. Prefer an explicit function or constructor and the smallest change that handles the actual requirement.
- Flag fragmented control flow, redundant intermediate containers, repeated conversions, and state that can be derived from existing values. Show the simpler equivalent. Fewer lines alone are not an improvement; do not replace readable code with dense comprehensions, nested ternaries, or helpers controlled by many flags.
- Keep shared facts in one authoritative place. Flag copied catalogs, defaults, band metadata, and preprocessing decisions that can drift between consumers. Preserve configuration precedence, explicit `false`/`null`/`[]`, lazy optional imports, and stored result compatibility when suggesting consolidation.

## Defensive code without a supported failure mode

- Flag fallback imports for hard dependencies, broad exception catches, silent defaults after failures, and retries without a specific recoverable condition. Optional dependencies can remain lazy; unexpected failures should propagate with their context.
- Flag repeated validation after the input has already been validated, `getattr`/`hasattr` on guaranteed fields, impossible `None` checks, and compatibility branches for unsupported versions. Check the declared types, callers, supported versions, and dependency requirements before claiming a branch is unnecessary.
- Preserve validation at external boundaries and handling for real failures such as missing data, invalid user configuration, or recoverable CUDA OOM. Do not trade correct error reporting or a supported behavior for shorter code.

## Plain prose

Apply these rules to added documentation, docstrings, error messages, and PR text. Preserve facts, numbers, citations, qualifications, and exact API names.

- Replace inflated wording with the concrete action: `leverage`/`utilize` → `use`, `delve` → `study`, `showcase`/`underscore` → `show`, `elucidate` → `explain`, `ameliorate` → `improve`, and `ships with` → `includes`.
- Flag filler such as `seamlessly`, `powerful`, `game-changing`, `pivotal`, `comprehensive`, `robust`, and `unlock` when no specific property is stated. Cut `Notably`, `Importantly`, `It is worth noting`, stacked hedges, and closing sentences that repeat the paragraph.
- Flag dramatic colon reveals, repeated em-dash asides, “not just X, but Y” pivots, invented `X-first` labels, and marketing claims that replace an explanation. State what the code does and why the reader needs to know.
- Prefer `setup` to a vague `protocol`, `format` to a vague `schema`, and a specific verb to catch-all `supports`. These are contextual prose preferences, not token bans: keep technical uses such as Pydantic schemas, `typing.Protocol`, robust statistics, published titles, and quoted text. Do not rename APIs to satisfy a word list.

## Security

- Ban new pickle-based persistence and deserialization, including `pickle`, `dill`, `cloudpickle`, `shelve`, Joblib persistence, `pandas.read_pickle`/`to_pickle`, and NumPy `allow_pickle=True`. Use JSON or Parquet for records, non-object NumPy arrays, and safetensors for tensors. Do not add writers that create more pickle artifacts or a fallback that restores legacy pickle metadata.
- Flag new `torch.save`/`torch.load` checkpoint paths and recommend safetensors. `weights_only=True` still uses an unpickler and is not a security sandbox. Changes maintaining existing checkpoint compatibility must keep explicit `weights_only=True`; reject `weights_only=False`, unsafe retry fallbacks, or allowlisting checkpoint-supplied globals just to make a load succeed. See [PyTorch's loading guidance](https://docs.pytorch.org/docs/stable/generated/torch.load).
- Reject unsafe YAML loaders and `eval`/`exec` on configuration or dataset content. Require `yaml.safe_load` or a verified `SafeLoader` subclass. The repository's `_UniqueKeyLoader` is such a subclass; do not flag its `yaml.load(..., Loader=_UniqueKeyLoader)` solely by name.
- Flag shell command interpolation, `os.system`, and `shell=True` with untrusted values. Prefer argument lists with `shell=False`, and check whether user-controlled arguments can be interpreted as options. A normal `subprocess.run([...])` call is not itself a vulnerability.
- Treat `trust_remote_code=True`, `torch.hub.load`, and configurable import targets as code execution. Require reviewed code sources and immutable commit pins for downloaded code; a trust flag or checksum alone does not establish that code is safe. Keep arbitrary model targets limited to trusted configuration.
- For archive extraction or paths derived from dataset metadata, check destination containment, absolute paths, `..`, and symlink escapes. Use `filter="data"` for tar extraction plus appropriate file-count and unpacked-size limits for untrusted archives. Do not confuse reading tar members with extracting them onto disk. See [Python's extraction guidance](https://docs.python.org/3.12/library/tarfile.html#extraction-filters).
- Flag disabled TLS verification, removed integrity checks, hardcoded credentials, and secrets or signed URLs written to logs. For dependency changes, check advisories against the actual resolved version before claiming a known vulnerability.
- In GitHub Actions, flag executing untrusted PR code with secrets or write tokens, including unsafe `pull_request_target`/`workflow_run` combinations, and interpolating PR titles or other attacker-controlled text into shell scripts. Prefer minimal token permissions and full commit SHA pins for external actions. See [GitHub's workflow security guidance](https://docs.github.com/en/actions/reference/security/secure-use).

## Useful review findings

- Write like a colleague leaving an inline comment. Usually one or two short sentences are enough: state the specific problem and the smallest useful fix. Add detail only when the failure path would otherwise be unclear. Apply the plain-prose rules above to your own comments.
- Anchor each finding to a changed line. For duplication, name the existing function; for a weak test, say what it fails to exercise. For a security claim, identify the untrusted input and dangerous operation, or state the explicit policy violation. Do not inflate a style preference into a correctness or security defect.
- Skip canned headings, bold labels, numbered mini-essays, praise, apologies, rhetorical questions, and phrases such as “It is important to note,” “This could potentially,” or “To ensure robustness.” Do not explain basic Python or repeat the code the reader can already see. Include a small suggested patch when it is clearer than prose.
- Group repeated instances and put correctness and security before style. Use the platform's required severity fields without repeating them in the comment body. Skip diff summaries, speculative hardening, unrelated cleanup, and claims about AI authorship. No finding is better than an invented one.

For example, write “`load_yaml` already rejects duplicate keys. Reuse it here so the two loaders cannot drift.” For a weak test, write “This replaces the parser with a mock, so it never exercises parsing. Pass a small YAML input through the real parser.”
