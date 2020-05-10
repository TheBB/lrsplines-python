import lrspline as lr
import numpy as np

### SETUP MESH PARAMETERS
p = np.array([3,3]) # quadratic functions (this is order, polynomial degree+1)
n = p + 3           # number of basis functions, start with 4x4 elements
n_refinements = 5   # number of edge refinements


### GENERATE THE ACTUAL MESH
surf = lr.LRSplineSurface(n[0], n[1], p[0], p[1])

for i in range(n_refinements):

    # first get all elements for all edges
    edge  = [e for e in surf.elements.edge('south')]
    edge += [e for e in surf.elements.edge('east')]
    edge += [e for e in surf.elements.edge('west')]
    edge += [e for e in surf.elements.edge('north')]

    # secondly; get all functions with support on these elements
    edge_basis = [basis for element in edge for basis in element.support()]
    surf.refine(edge_basis)

    # print some debug information during processing
    print(f'Done with refinement nr {i}')
    print(f'  Number of basis functions: {len(surf.basis)}')
    print(f'  Number of elements:        {len(surf.elements)}')


### WRITE OUTPUT RESULTS

# write LRSpline result to file
with open('edge_refined_lrspline.lr', 'wb') as myfile:
    surf.write(myfile)
    print(f'Results written to file {myfile.name}')

# dump a debug mesh file
with open('mesh.eps', 'wb') as myfile:
    surf.write_postscript(myfile)
    print(f'Results written to file {myfile.name}')

