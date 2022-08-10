# Download JSUT
if [ ! -d ./jsut_ver1.1 ]; then
    echo "JSUT Data Download ..."
    curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    unzip -o jsut_ver1.1.zip
    rm jsut_ver1.1.zip
    echo "JSUT download complete!"
else
    echo "JSUT data is exists."
fi

# Download JSUT-label
if [ ! -d ./jsut-lab ]; then
    echo "JSUT-label Data Download ..."
    curl -LO https://github.com/r9y9/jsut-lab/archive/v0.1.1.zip
    unzip -o v0.1.1.zip
    rm v0.1.1.zip
    cp -r jsut-lab-0.1.1/basic5000/lab jsut_ver1.1/basic5000/
    echo "JSUT-label download complete!"
else
    echo "JSUT-label data is exists."
fi
