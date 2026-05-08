## Assignment Checklist

### Setup & Environment
- [ ] Run `bash setup.sh` successfully
- [ ] Verify model weights downloaded (stories42M.pt)

### Core Implementation
- [ ] **llama.py**
- [ ] **rope.py**
- [ ] **optimizer.py**
- [ ] **addition_data_generation.py**
- [ ] **addition_run.py**

### Testing & Validation
- [ ] Pass `python sanity_check.py` ## Run this for causal mask 
- [ ] Pass `python optimizer_test.py` 
- [ ] Pass `python rope_test.py` 
- [ ] Generate coherent text with `python run_llama.py --option generate`
- [ ] Complete SST zero-shot prompting
- [ ] Complete CFIMDB zero-shot prompting  
- [ ] Complete data generation for addition llama.
- [ ] Complete five abalations. 
- [ ] Achieve target accuracy 100% using a very tiny llama. 
- [ ] Train your best model 

### Advanced Features (Optional - A+)
- [ ] Create the tiniest adding llama. 

### Submission Preparation
- [ ] Generate all required output files
- [ ] Validate submission with `python prepare_submit.py /path/to/project/ ANDREWID`
- [ ] Verify file size < 15MB
- [ ] Create zip file with proper ANDREWID structure
- [ ] Include optional report and feedback files
- [ ] Final submission check before Canvas upload