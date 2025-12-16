import os
import sys
from dotenv import load_dotenv

# --- QUAN TRỌNG: SỬA DÒNG IMPORT ---
# Vì umls_loader nằm trong folder 'pipeline', ta dùng dấu chấm để truy cập
# Cú pháp: from <tên_folder>.<tên_file> import <tên_class>
try:
    from pipeline.umls_loader import UMLSLoader
except ImportError:
    # Fallback: Nếu chạy lỗi, thử thêm folder hiện tại vào sys.path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))
    from umls_loader import UMLSLoader
# -----------------------------------

def run_test():
    # 1. Load biến môi trường từ file .env (nằm cùng cấp với file này)
    load_dotenv()
    
    api_key = os.getenv("UMLS_API_KEY")
    if not api_key:
        print("❌ LỖI: Chưa đọc được UMLS_API_KEY từ file .env")
        print("   Hãy chắc chắn file .env nằm ngay cạnh file test_umls_fix.py")
        return
    else:
        print(f"✅ Đã đọc được API Key: {api_key[:5]}******")

    print("\n=== ĐANG GỌI MODULE TỪ THƯ MỤC 'pipeline/' ===")
    
    # 2. Khởi tạo Class
    loader = UMLSLoader()
    
    if not loader.is_available():
        print("⚠️ Cảnh báo: Key có vẻ không hợp lệ hoặc lỗi mạng.")

    # 3. Test danh sách từ khóa
    terms_to_check = [
    # --- NHÓM LÂM SÀNG & BỆNH HỌC (SNOMEDCT_US, ICD10CM, MSH) ---
    "Fever",                # Kiểm tra SNOMEDCT_US, ICD10CM (Triệu chứng phổ biến)
    "Appendicitis",         # Kiểm tra ICD10CM (Bệnh học tiêu chuẩn)
    "Heart attack",         # Kiểm tra CHV (Consumer Health Vocab - Từ ngữ đời thường thay vì Myocardial Infarction)

    # --- NHÓM THUỐC & VẮC-XIN (RXNORM, ATC, CVX, MED-RT) ---
    "Metformin",            # Kiểm tra RXNORM (Tên hoạt chất thuốc)
    "Beta blocking agents", # Kiểm tra ATC (Nhóm thuốc/Lớp dược lý)
    "MMR vaccine",          # Kiểm tra CVX (Vắc-xin sởi-quai bị-rubella)
    "Mechanism of Action",  # Kiểm tra MED-RT (Khái niệm dược lý)

    # --- NHÓM XÉT NGHIỆM & THỦ THUẬT (LNC, ICD10PCS) ---
    "Hemoglobin A1c",       # Kiểm tra LNC (LOINC - Chỉ số xét nghiệm)
    "Appendectomy",         # Kiểm tra ICD10PCS (Mã phẫu thuật/thủ thuật)

    # --- NHÓM GIẢI PHẪU & SINH HỌC (FMA, GO, HPO) ---
    "Left ventricle",       # Kiểm tra FMA (Giải phẫu tim chi tiết)
    "HMG-CoA reductase",    # Kiểm tra GO (Gene Ontology - Chức năng phân tử)
    "Arachnodactyly",       # Kiểm tra HPO (Kiểu hình ngón tay nhện - Triệu chứng di truyền)

    # --- NHÓM GEN & UNG THƯ (NCI, HGNC, OMIM) ---
    "Glioblastoma",         # Kiểm tra NCI (Ung thư não - NCI Thesaurus rất mạnh về cái này)
    "TP53",                 # Kiểm tra HGNC (Tên chuẩn của Gien)
    "Marfan syndrome",      # Kiểm tra OMIM (Bệnh di truyền hiếm gặp)

    # --- NHÓM HÀNH CHÍNH & VẬT TƯ (HL7V3.0, HCPCS) ---
    "Patient",              # Kiểm tra HL7V3.0 (Vai trò trong y tế)
    "Wheelchair",           # Kiểm tra HCPCS (Thiết bị y tế/Xe lăn)
    "Walking"               # Kiểm tra SNOMEDCT_US (Hoạt động chức năng)
]

    print(f"\n{'TERM':<15} | {'ID (CUI)':<10} | {'NAME':<30} | {'STATUS'}")
    print("-" * 70)

    for term in terms_to_check:
        result = loader.get_best_match(term)
        
        if result:
            cui = result['umls_id']
            name = result['name']
            source = result.get('source', 'N/A')
            
            # Check lỗi C5958835
            if cui == "C5958835":
                status = "❌ RÁC"
            else:
                status = f"✅ OK ({source})"
                
            print(f"{term:<15} | {cui:<10} | {name[:28]:<30} | {status}")
        else:
            print(f"{term:<15} | {'Not Found'} | ...")

if __name__ == "__main__":
    run_test()

# ------------------------------------------------------------
# unified header — 2025-12-16.
# ------------------------------------------------------------