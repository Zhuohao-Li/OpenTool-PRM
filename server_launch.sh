export PROJECT_ENV=local
pip install chardet
python test_dependencies.py
python search_r1/search/rlab_search.py --top 3 --snippet_only
