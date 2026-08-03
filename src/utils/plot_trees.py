import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import re
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('models/model_optuna.cbm')
feat_names = model.feature_names_

SCALER = {
    'DENST':  {'mean': 1.0170046046046044,  'std': 0.007688568976846775},
    'HEMATT': {'mean': 101.39406072739406,   'std': 111.76485697517653},
    'RBO':    {'mean': 1586.1292425759093,   'std': 6980.803059995904},
    'WBCO':   {'mean': 2636.917257257257,    'std': 7524.9127452222565},
    'EC':     {'mean': 28.580013346680012,   'std': 83.73992469896548},
    'BACTS':  {'mean': 18487.615115115113,   'std': 30677.177086134197},
}
DISPLAY = {
    'DENST': 'Densidad', 'HEMATT': 'Hematies (tira)', 'RBO': 'RBC',
    'WBCO': 'WBC', 'EC': 'Cel. Epit.', 'BACTS': 'Bacterias',
    'SEXO_M': 'Mujer', 'LEUT_25': 'LEU=25', 'LEUT_75': 'LEU=75',
    'LEUT_500': 'LEU=500', 'NITT_1': 'Nitrito+',
    'PROTT_1': 'Proteina+', 'BACT_INFO_baja_1': 'Gram+',
    'BACT_INFO_baja_2': 'Gram mixto', 'BACT_INFO_baja_3': 'Sin info Gram',
    'EDAD_CATEGORICA_28-37': 'Edad 28-37',
}

def unscale(fname, threshold):
    if fname in SCALER:
        real = threshold * SCALER[fname]['std'] + SCALER[fname]['mean']
        if fname == 'DENST':
            return f"{real:.4f}"
        elif abs(real) >= 100:
            return f"{real:,.0f}"
        else:
            return f"{real:.1f}"
    return None

def parse_tree(model, tree_idx):
    graph = model.plot_tree(tree_idx=tree_idx)
    dot = graph.source
    nodes, edges = {}, {}
    # Join multiline DOT into single string, then parse with DOTALL
    for m in re.finditer(r'(\d+)\s*\[label="(.*?)".*?shape=(\w+)', dot, re.DOTALL):
        nid = int(m.group(1))
        label = m.group(2).replace('\n', '').strip()
        nodes[nid] = {'label': label, 'is_leaf': m.group(3) == 'rect'}
    for m in re.finditer(r'(\d+)\s*->\s*(\d+)\s*\[label=(\w+)', dot):
        edges.setdefault(int(m.group(1)), []).append((int(m.group(2)), m.group(3)))
    return nodes, edges

def get_splits_and_leaves(nodes, edges):
    splits = []
    queue = [0]
    for level in range(4):
        nid = queue[0]
        info = nodes[nid]
        parts = info['label'].split(', value>')
        fidx, thr = int(parts[0]), float(parts[1])
        fname = feat_names[fidx]
        dname = DISPLAY.get(fname, fname)
        rv = unscale(fname, thr)
        splits.append((fname, dname, rv if rv else None))
        next_queue = []
        for nid in queue:
            if nid in edges:
                for cid, el in edges[nid]:
                    next_queue.append(cid)
        queue = next_queue
    leaves = []
    for nid in queue:
        val = float(nodes[nid]['label'].replace('val = ', '').strip())
        leaves.append(val)
    return splits, leaves

def draw_tree_with_table(tree_idx):
    nodes, edg = parse_tree(model, tree_idx)
    splits, leaves = get_splits_and_leaves(nodes, edg)

    fig = plt.figure(figsize=(22, 16))

    # ── Top: split diagram ──
    ax = fig.add_axes([0.02, 0.35, 0.96, 0.58])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-5, 1)
    ax.axis('off')

    # Draw each level
    positions_by_level = {0: [8.0]}
    for level in range(4):
        fname, dname, real_thr = splits[level]
        if real_thr:
            label = f"{dname} > {real_thr}"
        else:
            label = f"{dname} = Si?"
        y = -level * 1.2
        current = positions_by_level[level]
        next_pos = []
        for px in current:
            ax.text(px, y, label, fontsize=14, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.45', fc='#BBDEFB', ec='#1565C0', lw=2.5), zorder=3)
            spread = 4.0 / (2 ** level)
            lx, rx = px - spread / 2, px + spread / 2
            cy = y - 1.2
            ax.plot([px, lx], [y - 0.35, cy + 0.35], color='#1565C0', lw=2, zorder=1)
            ax.plot([px, rx], [y - 0.35, cy + 0.35], color='#E65100', lw=2, zorder=1)
            mlx = (px + lx) / 2 - 0.1
            mrx = (px + rx) / 2 + 0.1
            mly = mry = (y + cy) / 2
            ax.text(mlx, mly, 'No', fontsize=10, ha='center', va='center',
                    color='#1565C0', fontweight='bold',
                    bbox=dict(fc='white', ec='none', alpha=0.85, boxstyle='round,pad=0.1'))
            ax.text(mrx, mry, 'Si', fontsize=10, ha='center', va='center',
                    color='#E65100', fontweight='bold',
                    bbox=dict(fc='white', ec='none', alpha=0.85, boxstyle='round,pad=0.1'))
            next_pos.extend([lx, rx])
        positions_by_level[level + 1] = next_pos

    ax.set_title(f'Arbol {tree_idx}  (de 101 arboles, profundidad = 4)',
                 fontsize=22, fontweight='bold', pad=20)

    # ── Bottom: leaf value strip ──
    ax2 = fig.add_axes([0.02, 0.05, 0.96, 0.27])
    ax2.set_xlim(-0.5, 16.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.axis('off')

    max_abs = max(abs(v) for v in leaves)
    for i, val in enumerate(leaves):
        x = i + 0.05
        w = 0.9
        if val > 0:
            intensity = abs(val) / max_abs
            r = 0.65 - 0.35 * intensity
            g = 0.84 - 0.14 * intensity
            b = 0.65 - 0.35 * intensity
            ec = '#2E7D32'
            txt = f"+{val:.3f}"
        else:
            intensity = abs(val) / max_abs
            r = 0.94
            g = 0.60 - 0.30 * intensity
            b = 0.60 - 0.30 * intensity
            ec = '#C62828'
            txt = f"{val:.3f}"

        rect = FancyBboxPatch((x, 0.8), w, 0.8,
                              boxstyle='round,pad=0.05',
                              facecolor=(r, g, b), edgecolor=ec, linewidth=2.5)
        ax2.add_patch(rect)
        ax2.text(i + 0.5, 1.2, txt, fontsize=13, ha='center', va='center',
                 fontweight='bold', color='black')

        # Path labels above each leaf
        path_parts = []
        for level in range(4):
            fname, dname, real_thr = splits[level]
            bit = (i >> (3 - level)) & 1
            short = dname.split('(')[0].strip()[:8]
            if real_thr:
                sym = '>' if bit == 1 else chr(0x2264)
                path_parts.append(f"{short}{sym}{real_thr}")
            else:
                path_parts.append(f"{short}={'Si' if bit==1 else 'No'}")
        ax2.text(i + 0.5, 2.1, '\n'.join(path_parts), fontsize=7,
                 ha='center', va='center', color='#555', family='monospace')

        ax2.text(i + 0.5, 0.55, f"Hoja {i}", fontsize=7, ha='center', va='center', color='#999')

    # Legend
    pos_p = mpatches.Patch(color='#A5D6A7', label='+ = UTI probable')
    neg_p = mpatches.Patch(color='#EF9A9A', label='- = UTI improbable')
    fig.legend(handles=[pos_p, neg_p], loc='lower center', ncol=2, fontsize=14,
               frameon=True, bbox_to_anchor=(0.5, 0.0))

    path = f'figures/catboost_tree_{tree_idx}_real.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()

for t in range(3):
    draw_tree_with_table(t)
