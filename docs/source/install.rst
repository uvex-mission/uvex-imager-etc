Installation
============

First, clone the uvex-imager-etc repository to your local machine.

.. code-block:: bash

   git clone https://github.com/uvex-mission/uvex-imager-etc.git 

...which will download the files into the uvex-imager-etc directory.

.. code-block:: bash

   cd uvex-imager-etc

Once this is complete, install the UVEX ETC:

.. code-block:: bash

   pip install -e .

This should make uvex_imager_etc importable anywhere. Use editable mode to be able to import the response files properly.

The UVEX ETC does not come pre-installed with the latest UVEX response curves as these are managed separately in uvex_response. To finish setting up the ETC, download the latest UVEX CALDB from the `UVEX website <https://uvex.caltech.edu/page/uvex-etc>`_ and place the unzipped, dated folder in uvex_imager_etc/response_files.

The ETC will automatically use the most recent dated and versioned CALDB folder it can find in response_files, though a specific release can be referenced by its folder name should you need to reproduce past results. Please check the website for updates before starting work.
