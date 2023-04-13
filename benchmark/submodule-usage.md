**The following takes pyjuice as an example.**

## Add a new submodule
```bash
git submodule add https://github.com/Juice-jl/pyjuice.git
```
This will update the `/.gitmodule` file and clone the submodule repo into the *current pwd*. The HEAD commit of submodule main(master) *at the time of adding* is tracked unless updated later.

<details>
<summary>About submodule file</summary>

`.gitmodule` in repo root contains config for all submodules. The command before generates:
```git
[submodule "benchmark/pyjuice"]
	path = benchmark/pyjuice
	url = https://github.com/Juice-jl/pyjuice.git
```
(this is the minimum profile and more can be configured)
  
`pyjuice` will be committed as a mode 160000 file with content:
```
Subproject commit f00f1743fe237091c78e7f786396d0cfb7f68ecf
```
(`f00f17` is the latest commit at the time of writing)

</details>

Any repo (public/private) can be added as a submodule of any repo (public/private), as long as you have proper access to the repos.

## Clone a repo with submodule(s)
```shell
git clone --recurse-submodules https://github.com/april-tools/cirkit.git
```

If you forgot to clone with submodules (or switched branch, etc.), run to fetch:
```shell
git submodule update --init --recursive
```
