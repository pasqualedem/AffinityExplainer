/* Step-through of AffEx on one real episode.
 *
 * Every image the stage shows was produced by running INSID3 and the explainer on one
 * 3-shot COCO episode of the class person, so the walk-through is the method rather than
 * an illustration of it. Aggregation is the uniform mean over levels the paper uses. The
 * one exception is the level strip: INSID3 matches at a single level, so those four
 * panels are that attribution redrawn from coarse to fine, and the page labels them as
 * illustrative.
 */
document.addEventListener('DOMContentLoaded', () => {
    const stage = document.getElementById('affex-anim-container');
    if (!stage) return;

    const caption = document.getElementById('afx-caption');
    const btnPrev = document.getElementById('afx-btn-previous');
    const btnNext = document.getElementById('afx-btn-next');
    const btnPlay = document.getElementById('afx-btn-play');
    const progress = document.getElementById('afx-progress');

    const show = (sel, on = true) =>
        stage.querySelectorAll(sel).forEach(el => el.classList.toggle('is-shown', on));
    const swap = (panel, key) => {
        stage.querySelectorAll(`[data-panel="${panel}"] img`).forEach(img =>
            img.classList.toggle('is-shown', img.dataset.frame === key));
    };
    // The callout lands on the attribution, so it follows the output cells.
    const supportState = (n, cls, on) =>
        stage.querySelectorAll(`.afx-final[data-support="${n}"]`).forEach(el => el.classList.toggle(cls, on));

    // Each step is a full description of the stage, so stepping backwards is just
    // replaying from the start rather than trying to undo animations.
    const steps = [
        {
            caption: 'One episode: a query of surfers riding a wave, and three support images of people, each with its object masked in red. The model segments this class from these three examples alone.',
            apply: () => {
                swap('query', 'plain');
                swap('shot1', 'plain');
                swap('shot2', 'plain');
                swap('shot3', 'plain');
            }
        },
        {
            caption: 'The model segments the query. That prediction is the region AffEx explains: <strong>which support pixels produced it?</strong>',
            apply: () => {
                swap('query', 'pred');
                swap('shot1', 'masked');
                swap('shot2', 'masked');
                swap('shot3', 'masked');
            }
        },
        {
            caption: 'The model already compares every query position with every support position, one score volume per feature level. Slicing those volumes at the selected region, averaging over it and normalising with a softmax gives one attribution map per level, for each shot. No gradient, no perturbation.',
            apply: () => {
                show('.afx-drop');
                show('.afx-scores');
                show('.afx-region-stage');
                show('.afx-levels');
                for (let shot = 1; shot <= 3; shot++) {
                    for (let lvl = 1; lvl <= 4; lvl++) swap(`level${lvl}_shot${shot}`, 'map');
                }
            }
        },
        {
            caption: 'The per-level maps are aggregated into a single attribution for each support image.',
            apply: () => {
                show('.afx-drop');
                show('.afx-scores');
                show('.afx-region-stage');
                show('.afx-levels');
                show('.afx-arrow');
                stage.querySelector('.afx-levels').classList.add('is-merging');
                stage.querySelector('.afx-merge').classList.add('is-flowing');
                show('.afx-out');
                swap('out1', 'map');
                swap('out2', 'map');
                swap('out3', 'map');
            }
        },
        {
            caption: 'Finally the support mask sets the sign, so pixels on the object count positively and pixels outside it negatively. The background stops competing with the object and the map takes its shape.',
            apply: () => {
                show('.afx-drop');
                show('.afx-scores');
                show('.afx-region-stage');
                show('.afx-levels');
                show('.afx-arrow');
                stage.querySelector('.afx-levels').classList.add('is-merging');
                show('.afx-out');
                swap('out1', 'map');
                swap('out2', 'map');
                swap('out3', 'map');
                show('.afx-mask-stage');
                show('.afx-final');
                swap('fin1', 'map');
                swap('fin2', 'map');
                swap('fin3', 'map');
            }
        },
        {
            caption: 'Read it back. Every shot is marked on its people and on nothing else: not the pitch, not the court, not the garden path. The frisbee group carries the most, the tennis player the sharpest peak. That is the question a practitioner has: <strong>which example is doing the work, and on what part of it?</strong>',
            apply: () => {
                show('.afx-drop');
                show('.afx-scores');
                show('.afx-region-stage');
                show('.afx-levels');
                show('.afx-arrow');
                stage.querySelector('.afx-levels').classList.add('is-merging');
                show('.afx-out');
                swap('out1', 'map');
                swap('out2', 'map');
                swap('out3', 'map');
                show('.afx-mask-stage');
                show('.afx-final');
                swap('fin1', 'map');
                swap('fin2', 'map');
                swap('fin3', 'map');
                supportState(2, 'is-dimmed', true);
                supportState(3, 'is-dimmed', true);
                supportState(1, 'is-lead', true);
                show('.afx-tag');
            }
        }
    ];

    function reset() {
        stage.querySelectorAll('.is-shown').forEach(el => el.classList.remove('is-shown'));
        stage.querySelector('.afx-levels').classList.remove('is-merging');
        stage.querySelector('.afx-merge').classList.remove('is-flowing');
        supportState(1, 'is-dimmed', false);
        supportState(1, 'is-lead', false);
        supportState(2, 'is-dimmed', false);
        supportState(3, 'is-dimmed', false);
    }

    // The query and the support set feed the model: draw those two wires from where the
    // boxes actually are, and redraw them whenever the layout moves.
    function drawWires() {
        const stageBox = stage.querySelector('.afx-stage');
        const model = stage.querySelector('.afx-model');
        const query = stage.querySelector('[data-panel="query"]');
        const supports = stage.querySelector('.afx-support-row');
        if (!stageBox || !model || !query || !supports) return;

        const base = stageBox.getBoundingClientRect();
        const box = el => {
            const r = el.getBoundingClientRect();
            return { x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height };
        };
        const m = box(model);
        const target = { x: m.x - 8, y: m.y + m.h / 2 };

        [['#afx-wire-query', query], ['#afx-wire-support', supports]].forEach(([sel, el]) => {
            const b = box(el);
            const from = { x: b.x + b.w + 6, y: b.y + b.h / 2 };
            const midX = from.x + (target.x - from.x) / 2;
            stage.querySelector(sel).setAttribute(
                'd', `M${from.x} ${from.y} C${midX} ${from.y}, ${midX} ${target.y}, ${target.x} ${target.y}`);
        });
    }

    if (window.ResizeObserver) {
        new ResizeObserver(drawWires).observe(stage);
    }
    window.addEventListener('resize', drawWires);
    window.addEventListener('load', drawWires);

    let index = 0;
    function render(target) {
        index = Math.max(0, Math.min(steps.length - 1, target));
        reset();
        for (let i = 0; i <= index; i++) steps[i].apply();
        // On a phone the stage shows only the part of the pipeline the step is about.
        const stageBox = stage.querySelector('.afx-stage');
        if (stageBox) stageBox.dataset.step = String(index);
        caption.innerHTML = steps[index].caption;
        btnPrev.disabled = index === 0;
        btnNext.disabled = index === steps.length - 1;
        progress.querySelectorAll('.afx-dot').forEach((d, i) =>
            d.classList.toggle('is-done', i <= index));
        drawWires();
    }

    let timer = null;
    function stopPlaying() {
        clearInterval(timer);
        timer = null;
        btnPlay.textContent = 'Play';
        btnPlay.classList.remove('is-primary');
    }
    btnPlay.addEventListener('click', () => {
        if (timer) return stopPlaying();
        btnPlay.textContent = 'Pause';
        btnPlay.classList.add('is-primary');
        if (index === steps.length - 1) render(0);
        timer = setInterval(() => {
            if (index === steps.length - 1) return stopPlaying();
            render(index + 1);
        }, 4200);
    });
    btnNext.addEventListener('click', () => { stopPlaying(); render(index + 1); });
    btnPrev.addEventListener('click', () => { stopPlaying(); render(index - 1); });

    steps.forEach(() => {
        const dot = document.createElement('span');
        dot.className = 'afx-dot';
        progress.appendChild(dot);
    });

    render(0);
});
