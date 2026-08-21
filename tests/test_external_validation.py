from tcc_ecg.external_validation import audit_wfdb_directory


def test_wfdb_header_audit_reads_leads_frequency_and_diagnoses(tmp_path):
    (tmp_path / "record.hea").write_text(
        "record 12 500 5000\n" + "\n".join(["record.dat 16 1000/mV 16 0 0 0 lead"] * 12) + "\n#Dx: 164889003,270492004\n",
        encoding="utf-8",
    )
    audit = audit_wfdb_directory(tmp_path)
    assert audit["records_found"] == 1
    assert audit["observed_leads"] == 12
    assert audit["observed_frequency"] == 500.0
    assert audit["diagnosis_counts"]["164889003"] == 1

