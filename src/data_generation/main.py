import sys
import os

# This is a bit of a hack to import the generator from the other repository.
# A better solution would be to make the generator a proper package.
sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "random_generador_facturas_pdf"
    )
)

from generator import generar_pdf  # noqa: E402


if __name__ == "__main__":
    # The output directory for the generated invoices.
    output_dir = os.path.dirname(__file__)
    num_invoices = 20  # Let's generate 20 invoices.

    print(
        f"Generating {num_invoices} individual invoice PDFs in "
        f"'{output_dir}'..."
    )

    # The generator script saves files to the current working directory.
    # So we change to the output directory before calling it.
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        # We want individual PDF files for each invoice.
        generar_pdf(n=num_invoices, individuales=True)
    finally:
        # Change back to the original directory.
        os.chdir(original_cwd)

    print("Done.")
