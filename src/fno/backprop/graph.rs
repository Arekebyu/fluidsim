#[derive(Clone, Copy, Debug)]
pub struct Variable {
    id: usize,
}

#[derive(Clone)]
pub enum Op {
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Sin(usize),
    ReLU(usize),
}

pub struct Node {
    val: f32,
    grad: f32,
    op: Option<Op>,
}

// arena allocation
#[derive(Default)]
pub struct Context {
    nodes: Vec<Node>,
}

impl Context {
    pub fn variable(&mut self, val: f32) -> Variable {
        let id = self.nodes.len();
        self.nodes.push(Node {
            val,
            grad: 0.0,
            op: None,
        });
        Variable { id }
    }

    /// make function for generic operations//

    pub fn add(&mut self, lhs: Variable, rhs: Variable) -> Variable {
        let id = self.nodes.len();
        let val = self.nodes[lhs.id].val + self.nodes[rhs.id].val;
        self.nodes.push(Node {
            val,
            grad: 0.0,
            op: Some(Op::Add(lhs.id, rhs.id)),
        });
        Variable { id }
    }

    pub fn sub(&mut self, lhs: Variable, rhs: Variable) -> Variable {
        let id = self.nodes.len();
        let val = self.nodes[lhs.id].val - self.nodes[rhs.id].val;
        self.nodes.push(Node {
            val,
            grad: 0.0,
            op: Some(Op::Sub(lhs.id, rhs.id)),
        });
        Variable { id }
    }

    pub fn mul(&mut self, lhs: Variable, rhs: Variable) -> Variable {
        let id = self.nodes.len();
        let val = self.nodes[lhs.id].val * self.nodes[rhs.id].val;
        self.nodes.push(Node {
            val,
            grad: 0.0,
            op: Some(Op::Mul(lhs.id, rhs.id)),
        });
        Variable { id }
    }
    pub fn sin(&mut self, inp: Variable) -> Variable {
        let id = self.nodes.len();
        let val = f32::sin(self.nodes[inp.id].val);
        self.nodes.push(Node {
            val,
            grad: 0.0,
            op: Some(Op::Sin(inp.id)),
        });
        Variable { id }
    }
    pub fn relu(&mut self, inp: Variable) -> Variable {
        let id = self.nodes.len();
        let val = {
            if self.nodes[inp.id].val > 0.0 {
                self.nodes[inp.id].val
            } else {
                0.0
            }
        };
        self.nodes.push(Node {
            val,
            grad: 0.0,
            op: Some(Op::ReLU(inp.id)),
        });
        Variable { id }
    }
    pub fn zero_grad(&mut self) {
        for var in self.nodes.iter_mut() {
            var.grad = 0.0;
        }
    }
    pub fn update_val(&mut self, var: Variable, new_val: f32) {
        self.nodes[var.id].val = new_val;
    }
    pub fn get_val(&mut self, var: Variable) -> f32 {
        self.nodes[var.id].val
    }
    pub fn get_grad(&mut self, var: Variable) -> f32 {
        self.nodes[var.id].grad
    }
    pub fn backward(&mut self, var: Variable) {
        self.nodes[var.id].grad = 1.0;

        for node_id in (0..self.nodes.len()).rev() {
            let node = &self.nodes[node_id];
            let grad = node.grad;
            let op = node.op.clone();
            if let Some(op) = op {
                match op {
                    Op::Add(l, r) => {
                        self.nodes[l].grad += grad;
                        self.nodes[r].grad += grad;
                    }
                    Op::Sub(l, r) => {
                        self.nodes[l].grad += grad;
                        self.nodes[r].grad -= grad;
                    }
                    Op::Mul(l, r) => {
                        self.nodes[l].grad += grad * self.nodes[r].val;
                        self.nodes[r].grad += grad * self.nodes[l].val;
                    }
                    Op::Sin(inp) => {
                        self.nodes[inp].grad += f32::cos(self.nodes[inp].val) * grad;
                    }
                    Op::ReLU(inp) => {
                        if self.nodes[inp].val > 0.0 {
                            self.nodes[inp].grad += grad;
                        }
                    }
                }
            }
        }
    }
}
