document.addEventListener('DOMContentLoaded', () => {
    
    // --- Grid Population ---
    const containers = document.querySelectorAll('.afx-tensor-face, .afx-tensor-depth-1, .afx-tensor-depth-2');
    containers.forEach(container => {
        if(container.children.length > 0) return;
        for(let i=0; i<25; i++) {
            let div = document.createElement('div');
            container.appendChild(div);
        }
    });

    // --- State Variables ---
    let step = 0;
    const caption = document.getElementById('afx-caption');
    const stage = document.querySelector('.afx-stage'); // Used for class toggling
    const btn = document.getElementById('afx-btn-next');
    const resetBtn = document.getElementById('afx-btn-reset');
    const phaseLabel = document.getElementById('afx-phase-label');

    // --- Steps Logic ---
    const steps = [
        {
            text: "Step 1: Feature Extraction. (Standard FSS Model)",
            action: () => {
                stage.classList.add('afx-state-features');
                phaseLabel.innerText = "Standard FSS Model";
                stage.classList.add('afx-phase-fss');
            }
        },
        {
            text: "Step 2: Dense Matching. 4D Correlation Tensors (Visually represented as thick 3D stacks).",
            action: () => stage.classList.add('afx-state-matching')
        },
        {
            text: "Step 3: ROI Averaging. Collapsing Tensors to 2D Maps over Query ROI. (AffEx Starts Here)",
            action: () => {
                stage.classList.add('afx-state-average');
                stage.classList.remove('afx-phase-fss');
                stage.classList.add('afx-phase-affex');
                phaseLabel.innerText = "Affinity Explainer (AffEx)";
            }
        },
        {
            text: "Step 4: Softmax Normalization. Red = Match, Blue = Mismatch.",
            action: () => stage.classList.add('afx-state-softmax')
        },
        {
            text: "Step 5: Feature Ablation Weights. Calculating layer importance (w).",
            action: () => stage.classList.add('afx-state-weights')
        },
        {
            text: "Step 6: Aggregation. Weighted summation into final maps.",
            action: () => stage.classList.add('afx-state-aggregate')
        },
        {
            text: "Step 7: Projection. Final Attribution Map highlights Black cat (Red) and marks Dog as irrelevant (Blue).",
            action: () => {
                stage.classList.add('afx-state-result');
                btn.innerText = "Done";
                btn.disabled = true;
            }
        }
    ];

    if(btn) {
        btn.addEventListener('click', () => {
            if(step < steps.length) {
                caption.innerText = steps[step].text;
                steps[step].action();
                step++;
            }
        });
    }

    if(resetBtn) {
        resetBtn.addEventListener('click', () => {
            step = 0;
            const classesToRemove = [
                'afx-state-features', 'afx-state-matching', 'afx-state-average',
                'afx-state-softmax', 'afx-state-weights', 'afx-state-aggregate',
                'afx-state-result', 'afx-phase-fss', 'afx-phase-affex'
            ];
            stage.classList.remove(...classesToRemove);
            
            caption.innerText = "Initialize: Query (Car) vs Support Set (Car, Plant)";
            phaseLabel.innerText = "";
            btn.innerText = "Next Step";
            btn.disabled = false;
        });
    }

    // --- FIXED SCALING LOGIC ---
    function scaleAnimation() {
        const container = document.getElementById('affex-anim-container');
        const scaler = document.querySelector('.afx-scaler');
        
        if (!container || !scaler) return;

        // Fixed design dimensions
        const designWidth = 1100; 
        const designHeight = 650;
        
        // Get the actual available width from the parent container
        const availableWidth = container.offsetWidth;
        
        // Calculate scale
        let scale = availableWidth / designWidth;
        
        // Optional: Cap scaling at 1.0 so it doesn't get blurry on huge screens
        if (scale > 1) scale = 1;

        // Apply Scale to the inner wrapper
        scaler.style.transform = `scale(${scale})`;
        
        // Set height of the outer container to match the scaled content
        container.style.height = `${designHeight * scale}px`;
    }

    window.addEventListener('resize', scaleAnimation);
    scaleAnimation();
    setTimeout(scaleAnimation, 200); 
});