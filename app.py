import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "Sticker Remover API ready"}


# ============================================================
# ROI VERSION (Variant 1)
# ============================================================
@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    try:
        # Lazy import (loads module + models the first time)
        from remove_sam_lama_fast import process_image

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Model import failed: {str(e)}"
            }
        )

    try:
        suffix = os.path.splitext(file.filename)[1] or ".webp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:

            tmp_in.write(await file.read())
            in_path = tmp_in.name
            out_path = tmp_out.name

        # Process ROI
        process_image(in_path, out_path, slno=1)

        # Read output image
        result = open(out_path, "rb").read()

        # Cleanup
        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result,
            media_type="image/webp",
            headers={"X-Status": "success"}
        )

    except Exception:
        import traceback
        # 100% guarantee clean JSON error
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "ROI processing failed",
                "trace": traceback.format_exc()
            }
        )


# ============================================================
# FULL IMAGE VERSION (Variant 2)
# ============================================================
@app.post("/process-full")
async def process_full_endpoint(file: UploadFile = File(...)):
    try:
        # Lazy import full variant
        from remove_sam_lama_fast import process_image_full

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Model import failed (full): {str(e)}"
            }
        )

    try:
        suffix = os.path.splitext(file.filename)[1] or ".webp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:

            tmp_in.write(await file.read())
            in_path = tmp_in.name
            out_path = tmp_out.name

        # Process full-image version
        process_image_full(in_path, out_path, slno=1)

        result = open(out_path, "rb").read()

        # Cleanup
        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result,
            media_type="image/webp",
            headers={"X-Status": "success"}
        )

    except Exception:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Full image processing failed",
                "trace": traceback.format_exc()
            }
        )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
