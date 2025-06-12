mkdir -p data

# Download DWUG en
echo "Downloading DWUG en..."
wget https://zenodo.org/records/14028531/files/dwug_en.zip?download=1
unzip dwug_en.zip?download=1
rm dwug_en.zip?download=1
mv dwug_en -t data