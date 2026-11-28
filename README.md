ML Studio

ML Studio is the interactive web prototype of the bachelor's thesis Explainable Machine-Learning Classification of Heart-Disease Status on a Public Benchmark Dataset with an Interactive Web Prototype (Syangtan Anugraha, Kyungdong University Global, 2026). It combines an educational machine-learning platform with the thesis research: four classifiers (Logistic Regression, Decision Tree, Random Forest, SVM) on the UCI Heart Disease Cleveland subset (303 records, 13 inputs).

Live site: https://frontend-1-nine-gilt.vercel.app/
Source code: https://github.com/anugraha435/ML-Studios

The tool is for research and education only. It is not a diagnostic or clinical decision-support system.

Area	What it contains
Learn ML	53 written lessons in six tracks, with equations (KaTeX), interactive figures and quizzes
Algorithms	Seven from-scratch, tested implementations with live controls: linear regression, logistic regression, decision tree, random forest, SVM, k-NN, k-means
Playground	Upload a CSV (or use a bundled sample), pick a target, train in a Web Worker, compare algorithms, score new records
Heart Disease Research	Overview, demonstration (real inference from the fitted models), measured performance, SHAP explainability, dataset, methodology, limitations
Documentation, About, global search (Ctrl/Cmd+K), light/dark theme	

Everything runs in the browser. An optional FastAPI service serves the same four fitted models over HTTP.

Honesty rules

Every figure on the Research pages is read from files produced by research/export_artifacts.py. Values that were not measured show "Not yet reported". Content carries one of four labels: Measured result, Actual model inference, Educational simulation, Illustrative example. The demo and prediction areas show a medical disclaimer. The model is a research demonstration trained on a historical US benchmark, not validated on any other population, and is not a medical product.

Layout
frontend/   React 18 + TypeScript + Vite + Tailwind (src/, tests/, e2e/)
backend/    FastAPI inference service (app/, tests/, models/heart_models.joblib, Dockerfile)
research/   Training/analysis code and the reference outputs (run_experiment.py, export_artifacts.py)
render.yaml Render blueprint for the API
Run locally
bash
# frontend (Node 20+)
cd frontend && npm ci && npm run dev          # http://localhost:5173

# backend (Python 3.13)
cd backend && pip install -r requirements-dev.txt
ALLOWED_ORIGINS=http://localhost:5173 uvicorn app.main:app --reload   # http://localhost:8000/docs
# then point the frontend at it (optional):  echo VITE_API_URL=http://localhost:8000 > frontend/.env.local

The demo works without the backend. When VITE_API_URL is set, a "server" mode appears that calls the API.

Test
bash
cd frontend && npx tsc --noEmit && npx vitest run     # 53 unit/parity tests
cd frontend && npm run build && npx vite preview --port 4173 &
cd frontend && node e2e/smoke.mjs                      # Playwright smoke test (needs playwright)
cd backend && python -m pytest -q                      # 15 API tests

The parity tests assert that the browser inference matches scikit-learn to 1e-9 for all four models, and that the from-scratch algorithms match scikit-learn fixtures.

Regenerate the research artifacts
bash
cd research && python3 -I run_experiment.py && python3 -I export_artifacts.py

This refits the models (seed 42), checks that the recomputed AUCs equal the reported ones, and rewrites frontend/src/data/research/*.json and backend/models/heart_models.joblib. Use the pinned versions in backend/requirements.txt; the joblib file is tied to scikit-learn 1.9.1.

Deploy

Frontend (Vercel or Netlify, static). Root directory frontend, build npm run build, output dist. vercel.json and public/_redirects provide the SPA fallback and security headers (CSP, nosniff, frame denial). Set VITE_API_URL to the API origin if you deploy the backend, and add that origin to the CSP connect-src in vercel.json.

Backend (Render, Docker). Create a Blueprint from render.yaml, then set ALLOWED_ORIGINS to the exact frontend origin (no wildcard). Health check: GET /health. Locally: docker build -t ml-studio-api backend && docker run -p 8000:8000 -e ALLOWED_ORIGINS=http://localhost:5173 ml-studio-api.

The API: GET /health, GET /api/v1/model-info, POST /api/v1/predict, POST /api/v1/explain/{rf|lr}. Inputs are validated with strict ranges and unknown fields are rejected. Requests are rate limited (RATE_LIMIT), CORS is an allow-list, and request bodies are not stored or logged.

Verification status (measured on 23 Nov 2026)
53 frontend tests, 15 backend tests and 427 end-to-end checks passed (74 routes, 73 internal links, no console errors, no horizontal overflow at 375 px).
npm audit --omit=dev and pip-audit report no known vulnerabilities.
Lighthouse 12 (mobile emulation, local preview server, no CDN compression): Home performance 84 / accessibility 100 / best practices 100 / SEO 100 (measured after the editorial redesign; the other pages below were measured before it); Learn 85 / 100 / 100 / 100; Research demo 86 / 100 / 100 / 100 (performance / accessibility / best practices / SEO).
Usability: an informal System Usability Scale questionnaire completed by 12 volunteers gave a mean score of 65.8 (SD 11.3, 95% CI 58.7 to 73.0). It is a small convenience sample, not a usability study with intended users (thesis Section 5.2.8).
Not done: the Docker image has not been built (no Docker daemon in the build environment), and the optional FastAPI server mode is not part of the deployed static site. No clinical validation or external validation has been done.
Known limitations
Cleveland benchmark only (303 records, 1980s US). No external validation, no prospective evaluation.
Test set is 61 records; model AUC intervals overlap heavily, so the ranking of the four models is not established.
Lifestyle questions (salt, fruit and vegetables, physical activity) are not model inputs.
In-browser training is capped (5 MB, 10,000 rows, 100 columns; SVM 3,000 training rows).
Progress is stored only in the browser's localStorage; there are no accounts.
Lessons were written for this project and have not had an external expert review.
Licence

MIT for the source code (see LICENSE). Third-party components are listed in THIRD_PARTY_NOTICES.md. The UCI Heart Disease data are credited to Janosi, Steinbrunn, Pfisterer and Detrano (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation) and are used under the UCI repository terms (CC BY 4.0).
