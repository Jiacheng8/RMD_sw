"""No source payload may be removed on a transfer/hash/readback failure."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import os
import json
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('storage_relocation',Path(__file__).parents[1]/'ops/h20-storage-relocate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.write_bytes(b'reference-data'*100)
        self.destination=self.root/'object/final';self.sha=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.control=patch.object(m,'CONTROL',self.root/'control');self.control.start()
    def tearDown(self):
        self.control.stop();self.temp.cleanup()
    def transfer(self,digest=None):
        return m.transfer(self.source,self.destination,self.source.stat().st_size,digest or self.sha,'file.json')
    def test_success_final_name_readback_precedes_receipt(self):
        result=self.transfer();self.assertTrue(result['final_named_SHA_readback'])
        self.assertEqual(self.destination.read_bytes(),self.source.read_bytes())
        self.assertTrue((self.root/'control/files/file.json').exists())
    def test_stale_partial_is_preserved_and_fresh_name_is_used(self):
        self.destination.parent.mkdir();stale=self.destination.with_name(self.destination.name+'.partial');stale.write_bytes(b'interrupted')
        self.transfer();self.assertEqual(stale.read_bytes(),b'interrupted')
        self.assertEqual(self.destination.read_bytes(),self.source.read_bytes())
    def test_wrong_source_hash_keeps_original_no_receipt(self):
        with self.assertRaises(AssertionError):self.transfer('0'*64)
        self.assertTrue(self.source.exists());self.assertFalse(self.destination.exists())
        self.assertFalse((self.root/'control/files/file.json').exists())
    def test_wrong_existing_object_never_overwritten(self):
        self.destination.parent.mkdir();self.destination.write_bytes(b'wrong')
        with self.assertRaises(AssertionError):self.transfer()
        self.assertTrue(self.source.exists());self.assertEqual(self.destination.read_bytes(),b'wrong')
    def test_final_readback_failure_retains_original(self):
        with patch.object(m,'sha',return_value='0'*64):
            with self.assertRaises(AssertionError):self.transfer()
        self.assertTrue(self.source.exists());self.assertFalse((self.root/'control/files/file.json').exists())

    def test_controlled_resume_reuses_completed_full_SHA_receipt(self):
        self.transfer()
        with patch.dict(os.environ,{'H20_REUSE_VERIFIED_OBJECTS':'1'}), patch.object(m,'sha',side_effect=AssertionError('unnecessary second read')):
            result=self.transfer()
        self.assertTrue(result['reused_previously_verified_object'])
    def test_later_object_change_forces_full_hash_and_keeps_source(self):
        result=self.transfer();self.destination.write_bytes(b'x'*self.source.stat().st_size)
        os.utime(self.destination,(result['verified_epoch']+10,result['verified_epoch']+10))
        with patch.dict(os.environ,{'H20_REUSE_VERIFIED_OBJECTS':'1'}):
            with self.assertRaises(AssertionError):self.transfer()
        self.assertTrue(self.source.exists())
    def test_receipt_for_other_hash_never_authorizes_reuse(self):
        self.transfer();p=self.root/'control/files/file.json';v=json.loads(p.read_text());v['sha256']='0'*64;p.write_text(json.dumps(v))
        with patch.dict(os.environ,{'H20_REUSE_VERIFIED_OBJECTS':'1'}),patch.object(m,'sha',return_value='0'*64):
            with self.assertRaises(AssertionError):self.transfer()

if __name__=='__main__':unittest.main()
