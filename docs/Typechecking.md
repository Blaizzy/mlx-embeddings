# Type Checking with Sorbet

The majority of the code in Homebrew is written in Ruby which is a dynamic
language. To avail the benefits of static type checking, we have set up Sorbet in
our codebase which provides the benefits of static type checking to dynamic languages
like Ruby. <br> [Sorbet's Documentation](https://sorbet.org/docs/overview) is a
good place to get started if you want to dive deeper into Sorbet and it's abilities.

## Sorbet elements in the Homebrew Codebase

The [`sorbet/`](https://github.com/Homebrew/brew/tree/master/Library/Homebrew/sorbet)
directory in `Library/Homebrew` consists of:

- The `rbi/` directory. It contains all Ruby Interface files, which help Sorbet to
learn about constants, ancestors, and methods defined in ways it doesn’t understand
natively. RBI files for all gems are auto-generated using
[Tapioca](https://github.com/Shopify/tapioca#tapioca). We can also create a RBI
file to help Sorbet understand dynamic definitions.
For example: Sorbet assumes that `Kernel` is not necessarily included in our modules
and classes, hence we use RBI files to explicitly include the Kernel Module. Here is an
[example](https://github.com/Homebrew/brew/blob/72419630b4658da31556a0f6ef1dfa633cf4fe4f/Library/Homebrew/sorbet/rbi/homebrew.rbi#L3-L5)
in our codebase.

- The `config` file. It is actually a newline-separated list of arguments to pass to
`srb tc`, the same as if they’d been passed at the command line. Arguments in the config
file are always passed first (if it exists), followed by arguments provided on the
command line. We use it ignore the `Library/Homebrew/vendor` directory, which
contains gem definitions which we do not wish to type check.

- The `files.yaml` file. It contains a list of every Ruby file in the codebase
divided into 3 strictness levels, false, true and strict. The `false` files only
report errors related to the syntax, constant resolution and correctness of the
method signatures, and not type errors. We use this file to override strictness
on a file-by-file basis. Our longtime goal is to move all `false` files to `true`
and start reporting type errors on those files as well. If you are making changes
that require adding a new ruby file, we would urge you to add it to `true` and work
out the resulting type errors. Read more about Sorbet's strictness levels
[here](https://sorbet.org/docs/static#file-level-granularity-strictness-levels).

## Using `brew typecheck`

When run without any arguments, `brew typecheck`, will run considering the strictness levels
set in the `files.yaml` file. However, when typecheck is run on a specific file
or directory, more errors may show up since Sorbet can not resolve constants defined
outside the scope of the specified file. These problems can be solved with RBI files.
Currently `brew typecheck` provides `quiet`, `--file`, `--dir` and `--ignore` options
but you can explore more options with `srb tc --help` and passing them with `srb tc`.

## Resolving Type Errors

Sorbet reports type errors along with an error reference code, which can be used
to look up more information on how to debug the error, or what causes the error in
the Sorbet documentation. Here is how we debug some common type errors:

* Using `T.reveal_type`. In files which are `true` or higher, if we wrap a variable
or method call in `T.reveal_type`, Sorbet will show us what type it thinks that
variable has in the output of `srb tc`. This is particularly useful when writing
[method signatures](https://sorbet.org/docs/sigs) and debugging. Make sure to
remove this line from your code before committing your changes, since this is
just a debugging tool.

* One of the most frequent errors that we've encountered is: `7003: Method does not exist.`
Since Ruby is a very dynamic language, methods can be defined in ways Sorbet cannot
see statically. In such cases, check if the method exists at runtime, if not, then
Sorbet has caught a future bug! But, it is also possible that even though a method
exists at runtime, Sorbet cannot see it. In such cases, we use `*.rbi` files.
Read more about RBI files [here](https://sorbet.org/docs/rbi).

* Since Sorbet does not automatically assume that Kernel is to be included in Modules,
we may encounter many errors while trying to use methods like `puts`, `ohai`, `odebug` et cetera.
A simple workaround for this would be to add an extra `include Kernel` line in the
respective RBI file.

* The tips above are very generic and apply to lots of cases. For some common gotchas
when using Sorbet, refer to the [Sorbet Error Reference](https://sorbet.org/docs/error-reference)
and [FAQ](https://sorbet.org/docs/faq).

## Method Signatures

Detailed explanation about why we use Method Signatures and its syntax can be found
[here](https://sorbet.org/docs/sigs). The only extra thing to keep in mind is that
we add method signatures to RBI files instead of the actual method definition in
the code. This way we preserve the original code structure and everything related to
Sorbet is kept within the `Library/Homebrew/sorbet` directory.
