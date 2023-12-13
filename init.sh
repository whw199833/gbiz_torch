pip install -r requirements.txt
TORCH=$(python3 -c 'import torch; print(torch.__version__)')
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git