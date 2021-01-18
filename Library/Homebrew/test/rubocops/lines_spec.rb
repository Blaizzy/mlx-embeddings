# typed: false
# frozen_string_literal: true

require "rubocops/lines"

describe RuboCop::Cop::FormulaAudit::Lines do
  subject(:cop) { described_class.new }

  context "when auditing deprecated special dependencies" do
    it "reports an offense when using depends_on :automake" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :automake
          ^^^^^^^^^^^^^^^^^^^^ :automake is deprecated. Usage should be \"automake\".
        end
      RUBY
    end

    it "reports an offense when using depends_on :autoconf" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :autoconf
          ^^^^^^^^^^^^^^^^^^^^ :autoconf is deprecated. Usage should be \"autoconf\".
        end
      RUBY
    end

    it "reports an offense when using depends_on :libtool" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :libtool
          ^^^^^^^^^^^^^^^^^^^ :libtool is deprecated. Usage should be \"libtool\".
        end
      RUBY
    end

    it "reports an offense when using depends_on :apr" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :apr
          ^^^^^^^^^^^^^^^ :apr is deprecated. Usage should be \"apr-util\".
        end
      RUBY
    end

    it "reports an offense when using depends_on :tex" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :tex
          ^^^^^^^^^^^^^^^ :tex is deprecated.
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::ClassInheritance do
  subject(:cop) { described_class.new }

  context "when auditing formula class inheritance" do
    it "reports an offense when not using spaces for class inheritance" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo<Formula
                  ^^^^^^^ Use a space in class inheritance: class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::Comments do
  subject(:cop) { described_class.new }

  context "when auditing comment text" do
    it "reports an offense when commented cmake calls exist" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          # system "cmake", ".", *std_cmake_args
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Please remove default template comments
        end
      RUBY
    end

    it "reports an offense when default template comments exist" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          # PLEASE REMOVE
          ^^^^^^^^^^^^^^^ Please remove default template comments
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
        end
      RUBY
    end

    it "reports an offense when `depends_on` is commented" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          # depends_on "foo"
          ^^^^^^^^^^^^^^^^^^ Commented-out dependency "foo"
        end
      RUBY
    end

    it "reports an offense if citation tags are present" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          # cite Howell_2009:
          ^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should not use `cite` comments
          # doi "10.111/222.x"
          ^^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should not use `doi` comments
          # tag "software"
          ^^^^^^^^^^^^^^^^ Formulae in homebrew/core should not use `tag` comments
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::AssertStatements do
  subject(:cop) { described_class.new }

  context "when auditing formula assertions" do
    it "reports an offense when assert ... include is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          assert File.read("inbox").include?("Sample message 1")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use `assert_match` instead of `assert ...include?`
        end
      RUBY
    end

    it "reports an offense when assert ... exist? is used without a negation" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          assert File.exist? "default.ini"
                 ^^^^^^^^^^^^^^^^^^^^^^^^^ Use `assert_predicate <path_to_file>, :exist?` instead of `assert File.exist? "default.ini"`
        end
      RUBY
    end

    it "reports an offense when assert ... exist? is used with a negation" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          assert !File.exist?("default.ini")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use `refute_predicate <path_to_file>, :exist?` instead of `assert !File.exist?("default.ini")`
        end
      RUBY
    end

    it "reports an offense when assert ... executable? is used without a negation" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          assert File.executable? f
                 ^^^^^^^^^^^^^^^^^^ Use `assert_predicate <path_to_file>, :executable?` instead of `assert File.executable? f`
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::OptionDeclarations do
  subject(:cop) { described_class.new }

  context "when auditing options" do
    it "reports an offense when `build.without?` is used in homebrew/core" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            build.without? "bar"
            ^^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should not use `build.without?`.
          end
        end
      RUBY
    end

    it "reports an offense when `build.with?` is used in homebrew/core" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            build.with? "bar"
            ^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should not use `build.with?`.
          end
        end
      RUBY
    end

    it "reports an offense when `build.without?` is used for a conditional dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "bar" if build.without?("baz")
                              ^^^^^^^^^^^^^^^^^^^^^ Use `:optional` or `:recommended` instead of `if build.without?("baz")`
        end
      RUBY
    end

    it "reports an offense when `build.without?` is used for a conditional dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "bar" if build.with?("baz")
                              ^^^^^^^^^^^^^^^^^^ Use `:optional` or `:recommended` instead of `if build.with?("baz")`
        end
      RUBY
    end

    it "reports an offense when `build.without?` is used with `unless`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return unless build.without? "bar"
                          ^^^^^^^^^^^^^^^^^^^^ Use if build.with? "bar" instead of unless build.without? "bar"
          end
        end
      RUBY
    end

    it "reports an offense when `build.with?` is used with `unless`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return unless build.with? "bar"
                          ^^^^^^^^^^^^^^^^^ Use if build.without? "bar" instead of unless build.with? "bar"
          end
        end
      RUBY
    end

    it "reports an offense when `build.with?` is negated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return if !build.with? "bar"
                      ^^^^^^^^^^^^^^^^^^ Don't negate 'build.with?': use 'build.without?'
          end
        end
      RUBY
    end

    it "reports an offense when `build.without?` is negated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return if !build.without? "bar"
                      ^^^^^^^^^^^^^^^^^^^^^ Don't negate 'build.without?': use 'build.with?'
          end
        end
      RUBY
    end

    it "reports an offense when a `build.without?` conditional is unnecessary" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return if build.without? "--without-bar"
                                     ^^^^^^^^^^^^^^^ Don't duplicate 'without': Use `build.without? \"bar\"` to check for \"--without-bar\"
          end
        end
      RUBY
    end

    it "reports an offense when a `build.with?` conditional is unnecessary" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return if build.with? "--with-bar"
                                  ^^^^^^^^^^^^ Don't duplicate 'with': Use `build.with? \"bar\"` to check for \"--with-bar\"
          end
        end
      RUBY
    end

    it "reports an offense when `build.include?` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def post_install
            return if build.include? "foo"
                      ^^^^^^^^^^^^^^^^^^^^ `build.include?` is deprecated
          end
        end
      RUBY
    end

    it "reports an offense when `def option` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'

          def options
          ^^^^^^^^^^^ Use new-style option definitions
            [["--bar", "desc"]]
          end
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::MpiCheck do
  subject(:cop) { described_class.new }

  context "when auditing MPI dependencies" do
    it "reports and corrects an offense when using depends_on \"mpich\" in homebrew/core" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "mpich"
          ^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should use 'depends_on "open-mpi"' instead of 'depends_on "mpich"'.
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "open-mpi"
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::SafePopenCommands do
  subject(:cop) { described_class.new }

  context "when auditing popen commands" do
    it "reports and corrects `Utils.popen_read` usage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read "foo"
            ^^^^^^^^^^^^^^^^^^^^^^ Use `Utils.safe_popen_read` instead of `Utils.popen_read`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read "foo"
          end
        end
      RUBY
    end

    it "reports and corrects `Utils.popen_write` usage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_write "foo"
            ^^^^^^^^^^^^^^^^^^^^^^^ Use `Utils.safe_popen_write` instead of `Utils.popen_write`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write "foo"
          end
        end
      RUBY
    end

    it "does not report an offense when `Utils.popen_read` is used in a test block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install; end
          test do
            Utils.popen_read "foo"
          end
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::ShellVariables do
  subject(:cop) { described_class.new }

  context "when auditing shell variables" do
    it "reports and corrects unexpanded shell variables in `Utils.popen`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen "SHELL=bash foo"
                        ^^^^^^^^^^^^^^^^ Use `Utils.popen({ "SHELL" => "bash" }, "foo")` instead of `Utils.popen "SHELL=bash foo"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen { "SHELL" => "bash" }, "foo"
          end
        end
      RUBY
    end

    it "reports and corrects unexpanded shell variables in `Utils.safe_popen_read`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read "SHELL=bash foo"
                                  ^^^^^^^^^^^^^^^^ Use `Utils.safe_popen_read({ "SHELL" => "bash" }, "foo")` instead of `Utils.safe_popen_read "SHELL=bash foo"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read { "SHELL" => "bash" }, "foo"
          end
        end
      RUBY
    end

    it "reports and corrects unexpanded shell variables in `Utils.safe_popen_write`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write "SHELL=bash foo"
                                   ^^^^^^^^^^^^^^^^ Use `Utils.safe_popen_write({ "SHELL" => "bash" }, "foo")` instead of `Utils.safe_popen_write "SHELL=bash foo"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write { "SHELL" => "bash" }, "foo"
          end
        end
      RUBY
    end

    it "reports and corrects unexpanded shell variables while preserving string interpolation" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen "SHELL=bash \#{bin}/foo"
                        ^^^^^^^^^^^^^^^^^^^^^^^ Use `Utils.popen({ "SHELL" => "bash" }, "\#{bin}/foo")` instead of `Utils.popen "SHELL=bash \#{bin}/foo"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen { "SHELL" => "bash" }, "\#{bin}/foo"
          end
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::LicenseArrays do
  subject(:cop) { described_class.new }

  context "when auditing license arrays" do
    it "reports no offenses for license strings" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license "MIT"
        end
      RUBY
    end

    it "reports no offenses for license symbols" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license :public_domain
        end
      RUBY
    end

    it "reports no offenses for license hashes" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", "0BSD"]
        end
      RUBY
    end

    it "reports and corrects use of a license array" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license ["MIT", "0BSD"]
          ^^^^^^^^^^^^^^^^^^^^^^^ Use `license any_of: ["MIT", "0BSD"]` instead of `license ["MIT", "0BSD"]`
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", "0BSD"]
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::Licenses do
  subject(:cop) { described_class.new }

  context "when auditing licenses" do
    it "reports no offenses for license strings" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license "MIT"
        end
      RUBY
    end

    it "reports no offenses for license symbols" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license :public_domain
        end
      RUBY
    end

    it "reports no offenses for license hashes" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", "0BSD"]
        end
      RUBY
    end

    it "reports no offenses for license exceptions" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license "MIT" => { with: "LLVM-exception" }
        end
      RUBY
    end

    it "reports no offenses for multiline nested license hashes" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: [
            "MIT",
            all_of: ["0BSD", "Zlib"],
          ]
        end
      RUBY
    end

    it "reports no offenses for multiline nested license hashes with exceptions" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: [
            "MIT",
            all_of: ["0BSD", "Zlib"],
            "GPL-2.0-only" => { with: "LLVM-exception" },
          ]
        end
      RUBY
    end

    it "reports an offense for nested license hashes on a single line" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", all_of: ["0BSD", "Zlib"]]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Split nested license declarations onto multiple lines
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::PythonVersions do
  subject(:cop) { described_class.new }

  context "when auditing Python versions" do
    it "reports no offenses for Python with no dependency" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            puts "python@3.8"
          end
        end
      RUBY
    end

    it "reports no offenses for unversioned Python references" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python"
          end
        end
      RUBY
    end

    it "reports no offenses for Python with no version" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3"
          end
        end
      RUBY
    end

    it "reports no offenses when a Python reference matches its dependency" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.9"
          end
        end
      RUBY
    end

    it "reports no offenses when a Python reference matches its dependency without `@`" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.9"
          end
        end
      RUBY
    end

    it "reports no offenses when a Python reference matches its two-digit dependency" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.10"

          def install
            puts "python@3.10"
          end
        end
      RUBY
    end

    it "reports no offenses when a Python reference matches its two-digit dependency without `@`" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.10"

          def install
            puts "python3.10"
          end
        end
      RUBY
    end

    it "reports and corrects Python references with mismatched versions" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.8"
                 ^^^^^^^^^^^^ References to `python@3.8` should match the specified python dependency (`python@3.9`)
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.9"
          end
        end
      RUBY
    end

    it "reports and corrects Python references with mismatched versions without `@`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.8"
                 ^^^^^^^^^^^ References to `python3.8` should match the specified python dependency (`python3.9`)
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.9"
          end
        end
      RUBY
    end

    it "reports and corrects Python references with mismatched two-digit versions" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python@3.10"
                 ^^^^^^^^^^^^^ References to `python@3.10` should match the specified python dependency (`python@3.11`)
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python@3.11"
          end
        end
      RUBY
    end

    it "reports and corrects Python references with mismatched two-digit versions without `@`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python3.10"
                 ^^^^^^^^^^^^ References to `python3.10` should match the specified python dependency (`python3.11`)
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python3.11"
          end
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAudit::Miscellaneous do
  subject(:cop) { described_class.new }

  context "when auditing formula miscellany" do
    it "reports an offense for unneeded `FileUtils` usage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          FileUtils.mv "hello"
          ^^^^^^^^^^^^^^^^^^^^ Don\'t need \'FileUtils.\' before mv
        end
      RUBY
    end

    it "reports an offense for long `inreplace` block variable names" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          inreplace "foo" do |longvar|
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \"inreplace <filenames> do |s|\" is preferred over \"|longvar|\".
            somerandomCall(longvar)
          end
        end
      RUBY
    end

    it "reports an offense for invalid `rebuild` numbers" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          bottle do
            rebuild 0
            ^^^^^^^^^ 'rebuild 0' should be removed
            sha256 "fe0679b932dd43a87fd415b609a7fbac7a069d117642ae8ebaac46ae1fb9f0b3" => :sierra
          end
        end
      RUBY
    end

    it "reports an offense when `OS.linux?` is used in homebrew/core" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          bottle do
            if OS.linux?
               ^^^^^^^^^ Don\'t use OS.linux?; homebrew/core only supports macOS
              nil
            end
            sha256 "fe0679b932dd43a87fd415b609a7fbac7a069d117642ae8ebaac46ae1fb9f0b3" => :sierra
          end
        end
      RUBY
    end

    it "reports an offense when a useless `fails_with :llvm` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          bottle do
            sha256 "fe0679b932dd43a87fd415b609a7fbac7a069d117642ae8ebaac46ae1fb9f0b3" => :sierra
          end
          fails_with :llvm do
          ^^^^^^^^^^^^^^^^ 'fails_with :llvm' is now a no-op so should be removed
            build 2335
            cause "foo"
          end
        end
      RUBY
    end

    it "reports an offense when `def test` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'

          def test
          ^^^^^^^^ Use new-style test definitions (test do)
            assert_equals "1", "1"
          end
        end
      RUBY
    end

    it "reports an offense when `skip_clean` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          skip_clean :all
          ^^^^^^^^^^^^^^^ `skip_clean :all` is deprecated; brew no longer strips symbols. Pass explicit paths to prevent Homebrew from removing empty folders.
        end
      RUBY
    end

    it "reports an offense when `build.universal?` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          if build.universal?
             ^^^^^^^^^^^^^^^^ macOS has been 64-bit only since 10.6 so build.universal? is deprecated.
             "foo"
          end
        end
      RUBY
    end

    it "reports no offenses when `build.universal?` is used in an exempt formula" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/wine.rb")
        class Wine < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          if build.universal?
             "foo"
          end
        end
      RUBY
    end

    it "reports an offense when `ENV.universal_binary` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          if build?
             ENV.universal_binary
             ^^^^^^^^^^^^^^^^^^^^ macOS has been 64-bit only since 10.6 so ENV.universal_binary is deprecated.
          end
        end
      RUBY
    end

    it "reports no offenses when `ENV.universal_binary` is used in an exempt formula" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/wine.rb")
        class Wine < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          if build?
            ENV.universal_binary
          end
        end
      RUBY
    end

    it "reports an offense when `install_name_tool` is called" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "install_name_tool", "-id"
                 ^^^^^^^^^^^^^^^^^^^ Use ruby-macho instead of calling "install_name_tool"
        end
      RUBY
    end

    it "reports an offense when `npm install` is called without Language::Node arguments" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "npm", "install"
          ^^^^^^^^^^^^^^^^^^^^^^^ Use Language::Node for npm install args
        end
      RUBY
    end

    it "reports no offenses when `npm install` is called without Language::Node arguments in an exempt formula" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/kibana@4.4.rb")
        class KibanaAT44 < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "npm", "install"
        end
      RUBY
    end

    it "reports an offense when `depends_on` is called with an instance" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on FOO::BAR.new
                     ^^^^^^^^^^^^ `depends_on` can take requirement classes instead of instances
        end
      RUBY
    end

    it "reports an offense when `Dir` is called without a globbing argument" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          rm_rf Dir["src/{llvm,test,librustdoc,etc/snapshot.pyc}"]
          rm_rf Dir["src/snapshot.pyc"]
                    ^^^^^^^^^^^^^^^^^^ Dir(["src/snapshot.pyc"]) is unnecessary; just use "src/snapshot.pyc"
        end
      RUBY
    end

    it "reports an offense when executing a system command for which there is a Ruby FileUtils equivalent" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "mkdir", "foo"
                 ^^^^^^^ Use the `mkdir` Ruby method instead of `system "mkdir", "foo"`
        end
      RUBY
    end

    it "reports an offense when top-level functions are defined outside of a class body" do
      expect_offense(<<~RUBY)
        def test
        ^^^^^^^^ Define method test in the class body, not at the top-level
           nil
        end
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
        end
      RUBY
    end

    it 'reports an offense when `man+"man8"` is used' do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            man1.install man+"man8" => "faad.1"
                             ^^^^^^ "man+"man8"" should be "man8"
          end
        end
      RUBY
    end

    it "reports an offense when a hard-coded `gcc` is referenced" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "/usr/bin/gcc", "foo"
                   ^^^^^^^^^^^^^^ Use "#{ENV.cc}" instead of hard-coding "gcc"
          end
        end
      RUBY
    end

    it "reports an offense when a hard-coded `g++` is referenced" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "/usr/bin/g++", "-o", "foo", "foo.cc"
                   ^^^^^^^^^^^^^^ Use "#{ENV.cxx}" instead of hard-coding "g++"
          end
        end
      RUBY
    end

    it "reports an offense when a hard-coded `llvm-g++` is set as COMPILER_PATH" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            ENV["COMPILER_PATH"] = "/usr/bin/llvm-g++"
                                   ^^^^^^^^^^^^^^^^^^^ Use "#{ENV.cxx}" instead of hard-coding "llvm-g++"
          end
        end
      RUBY
    end

    it "reports an offense when a hard-coded `gcc` is set as COMPILER_PATH" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            ENV["COMPILER_PATH"] = "/usr/bin/gcc"
                                   ^^^^^^^^^^^^^^ Use \"\#{ENV.cc}\" instead of hard-coding \"gcc\"
          end
        end
      RUBY
    end

    it "reports an offense when the formula path shortcut `man` could be used" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            mv "#{share}/man", share
                        ^^^^ "#{share}/man" should be "#{man}"
          end
        end
      RUBY
    end

    it "reports an offense when the formula path shortcut `libexec` could be used" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            mv "#{prefix}/libexec", share
                         ^^^^^^^^ "#{prefix}/libexec" should be "#{libexec}"
          end
        end
      RUBY
    end

    it "reports an offense when the formula path shortcut `info` could be used" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "./configure", "--INFODIR=#{prefix}/share/info"
                                                      ^^^^^^^^^^^ "#{prefix}/share/info" should be "#{info}"
          end
        end
      RUBY
    end

    it "reports an offense when the formula path shortcut `man8` could be used" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "./configure", "--MANDIR=#{prefix}/share/man/man8"
                                                     ^^^^^^^^^^^^^^^ "#{prefix}/share/man/man8" should be "#{man8}"
          end
        end
      RUBY
    end

    it "reports an offense when unvendored lua modules are used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "lpeg" => :lua51
                               ^^^^^^ lua modules should be vendored rather than use deprecated `depends_on \"lpeg\" => :lua51`
        end
      RUBY
    end

    it "reports an offense when `export` is used to set environment variables" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "export", "var=value"
                 ^^^^^^^^ Use ENV instead of invoking 'export' to modify the environment
        end
      RUBY
    end

    it "reports an offense when dependencies with invalid options are used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "foo" => "with-bar"
                              ^^^^^^^^^^ Dependency foo should not use option with-bar
        end
      RUBY
    end

    it "reports an offense when dependencies with invalid options are used in an array" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "httpd" => [:build, :test]
          depends_on "foo" => [:optional, "with-bar"]
                                          ^^^^^^^^^^ Dependency foo should not use option with-bar
          depends_on "icu4c" => [:optional, "c++11"]
                                            ^^^^^^^ Dependency icu4c should not use option c++11
        end
      RUBY
    end

    it "reports an offense when `build.head?` could be used instead of checking `version`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          if version == "HEAD"
             ^^^^^^^^^^^^^^^^^ Use 'build.head?' instead of inspecting 'version'
            foo()
          end
        end
      RUBY
    end

    it "reports an offense when `ARGV.include? (--HEAD)` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          test do
            head = ARGV.include? "--HEAD"
                   ^^^^ Use build instead of ARGV to check options
                   ^^^^^^^^^^^^^^^^^^^^^^ Use "if build.head?" instead
          end
        end
      RUBY
    end

    it "reports an offense when `needs :openmp` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          needs :openmp
          ^^^^^^^^^^^^^ 'needs :openmp' should be replaced with 'depends_on \"gcc\"'
        end
      RUBY
    end

    it "reports an offense when `MACOS_VERSION` is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          test do
            version = MACOS_VERSION
                      ^^^^^^^^^^^^^ Use MacOS.version instead of MACOS_VERSION
          end
        end
      RUBY
    end

    it "reports an offense when `build.with?` is used for a conditional dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "foo" if build.with? "foo"
          ^^^^^^^^^^^^^^^^ Replace depends_on "foo" if build.with? "foo" with depends_on "foo" => :optional
        end
      RUBY
    end

    it "reports an offense when `build.without?` is used for a negated conditional dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :foo unless build.without? "foo"
          ^^^^^^^^^^^^^^^ Replace depends_on :foo unless build.without? "foo" with depends_on :foo => :recommended
        end
      RUBY
    end

    it "reports an offense when `build.include?` is used for a negated conditional dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :foo unless build.include? "without-foo"
          ^^^^^^^^^^^^^^^ Replace depends_on :foo unless build.include? "without-foo" with depends_on :foo => :recommended
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAuditStrict::MakeCheck do
  subject(:cop) { described_class.new }

  let(:path) { Tap::TAP_DIRECTORY/"homebrew/homebrew-core" }

  before do
    path.mkpath
    (path/"style_exceptions").mkpath
  end

  def setup_style_exceptions
    (path/"style_exceptions/make_check_allowlist.json").write <<~JSON
      [ "bar" ]
    JSON
  end

  it "reports an offense when formulae in homebrew/core run build-time checks" do
    setup_style_exceptions

    expect_offense(<<~RUBY, "#{path}/Formula/foo.rb")
      class Foo < Formula
        desc "foo"
        url 'https://brew.sh/foo-1.0.tgz'
        system "make", "-j1", "test"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core (except e.g. cryptography, libraries) should not run build-time checks
      end
    RUBY
  end

  it "reports no offenses when exempted formulae in homebrew/core run build-time checks" do
    setup_style_exceptions

    expect_no_offenses(<<~RUBY, "#{path}/Formula/bar.rb")
      class Bar < Formula
        desc "bar"
        url 'https://brew.sh/bar-1.0.tgz'
        system "make", "-j1", "test"
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAuditStrict::ShellCommands do
  subject(:cop) { described_class.new }

  context "when auditing shell commands" do
    it "reports and corrects an offense when `system` arguments should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            system "foo bar"
                   ^^^^^^^^^ Separate `system` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            system "foo", "bar"
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `system` arguments with string interpolation should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            system "\#{bin}/foo bar"
                   ^^^^^^^^^^^^^^^^ Separate `system` commands into `\"\#{bin}/foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            system "\#{bin}/foo", "bar"
          end
        end
      RUBY
    end

    it "reports no offenses when `system` with metacharacter arguments are called" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            system "foo bar > baz"
          end
        end
      RUBY
    end

    it "reports no offenses when trailing arguments to `system` are unseparated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            system "foo", "bar baz"
          end
        end
      RUBY
    end

    it "reports no offenses when `Utils.popen` arguments are unseparated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen("foo bar")
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `Utils.popen_read` arguments are unseparated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo bar")
                             ^^^^^^^^^ Separate `Utils.popen_read` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo", "bar")
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `Utils.safe_popen_read` arguments are unseparated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read("foo bar")
                                  ^^^^^^^^^ Separate `Utils.safe_popen_read` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read("foo", "bar")
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `Utils.popen_write` arguments are unseparated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_write("foo bar")
                              ^^^^^^^^^ Separate `Utils.popen_write` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_write("foo", "bar")
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `Utils.safe_popen_write` arguments are unseparated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write("foo bar")
                                   ^^^^^^^^^ Separate `Utils.safe_popen_write` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write("foo", "bar")
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `Utils.popen_read` arguments with interpolation are unseparated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("\#{bin}/foo bar")
                             ^^^^^^^^^^^^^^^^ Separate `Utils.popen_read` commands into `\"\#{bin}/foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("\#{bin}/foo", "bar")
          end
        end
      RUBY
    end

    it "reports no offenses when `Utils.popen_read` arguments with metacharacters are unseparated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo bar > baz")
          end
        end
      RUBY
    end

    it "reports no offenses when trailing arguments to `Utils.popen_read` are unseparated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo", "bar baz")
          end
        end
      RUBY
    end

    it "reports and corrects an offense when `Utils.popen_read` arguments are unseparated after a shell variable" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read({ "SHELL" => "bash"}, "foo bar")
                                                   ^^^^^^^^^ Separate `Utils.popen_read` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read({ "SHELL" => "bash"}, "foo", "bar")
          end
        end
      RUBY
    end
  end
end
