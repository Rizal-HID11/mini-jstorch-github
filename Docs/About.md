# Mini-JSTorch — Technical Information

---

## General Information

- **Project Name:** mini-jstorch  
- **Internal Name:** JST (JST-orch)

> Note:  
> Early versions of JST do not strictly follow semantic versioning conventions  
> (e.g. `0.0.1` for patches, `0.1.0` for minor releases, `1.0.0` for major releases).  
> This inconsistency reflects the early learning and experimental phase of the project.

---

## 1. Engine Architecture Limitations (JST Core)

This section outlines the known structural weaknesses of the JST engine.  
Although the architecture may appear complex, it is currently sensitive and tightly coupled.

### Identified Limitations

- **High dependency on Utilities**  
  Every core class depends directly on the Utilities module, which is defined at the top of the `jstorch.js` file. This creates strong coupling across the engine.

- **Limited Tensor dimensionality**  
  Tensor implementations currently support only two dimensions.  
  Extending support to higher-dimensional tensors would require significant architectural changes due to the existing complexity.

- **Uneven class complexity**  
  New or recently modified classes often become significantly more complex than others, leading to inconsistency in maintainability and internal design balance.

---

## 2. Rationale Behind the `fu_` Utilities

This section explains why the `fu_` utilities were introduced despite the existence of internal Utilities.

### Issues with Internal Utilities

- The Utilities defined at the beginning of `jstorch.js` are **internal engine helpers**, not intended for direct user interaction.

- These Utilities are heavily reused across multiple core classes.  
  Any modification to a utility function may trigger **cascading (domino) errors** throughout the engine due to tight dependencies.

- Some utility functions intentionally diverge from standard or expected formulas.  
  For example:
  - Expected formula:  
    `Param1 - Param4 * Param3`
  - Internal Utilities implementation:  
    `Param1 - Param2 * Param3 + Param4`

  This behavior exists because internal Utilities are optimized for class-level computations, not for user-facing correctness or predictability.

### Purpose of `fu_` Utilities

The `fu_` utilities were designed to improve the **user experience** by providing:

- Predictable and correct computational behavior
- User-friendly and stable helper functions
- Isolation from internal engine changes
- Reduced risk of incorrect outputs and dependency-based cascading errors

In short, `fu_` exists to ensure safety, clarity, and consistency for end users of Mini-JSTorch.

---

## 3. SJK (Shortcut JST Keywords) Reference

This section lists commonly used abbreviations and keywords within the mini-jstorch ecosystem.

**Format:** `"KEYWORD" : "Full Name / Meaning"`

- `"JST"` : JSTorch  
- `"fu"` : For User / User-Friendly  
- `"fun"` : Function  
- `"Dummy"` : Experimental  
- `"Exp"` : Restricted experimental entity  
- `"msg"` : Message, comment, warning, announcement  
- `"donot"` : Do not / Don't  
// add more.

---
