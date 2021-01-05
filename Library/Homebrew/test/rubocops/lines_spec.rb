# typed: false
# frozen_string_literal: true

require "rubocops/lines"

describe RuboCop::Cop::FormulaAudit::Lines do
  subject(:cop) { described_class.new }

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

describe RuboCop::Cop::FormulaAudit::ClassInheritance do
  subject(:cop) { described_class.new }

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

describe RuboCop::Cop::FormulaAudit::Comments do
  subject(:cop) { described_class.new }

  context "When auditing formula" do
    it "commented cmake call" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          # system "cmake", ".", *std_cmake_args
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Please remove default template comments
        end
      RUBY
    end

    it "default template comments" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          # PLEASE REMOVE
          ^^^^^^^^^^^^^^^ Please remove default template comments
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
        end
      RUBY
    end

    it "commented out depends_on" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          # depends_on "foo"
          ^^^^^^^^^^^^^^^^^^ Commented-out dependency "foo"
        end
      RUBY
    end

    it "citation tags" do
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

  it "assert ...include usage" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        desc "foo"
        url 'https://brew.sh/foo-1.0.tgz'
        assert File.read("inbox").include?("Sample message 1")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use `assert_match` instead of `assert ...include?`
      end
    RUBY
  end

  it "assert ...exist? without a negation" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        desc "foo"
        url 'https://brew.sh/foo-1.0.tgz'
        assert File.exist? "default.ini"
               ^^^^^^^^^^^^^^^^^^^^^^^^^ Use `assert_predicate <path_to_file>, :exist?` instead of `assert File.exist? "default.ini"`
      end
    RUBY
  end

  it "assert ...exist? with a negation" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        desc "foo"
        url 'https://brew.sh/foo-1.0.tgz'
        assert !File.exist?("default.ini")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use `refute_predicate <path_to_file>, :exist?` instead of `assert !File.exist?("default.ini")`
      end
    RUBY
  end

  it "assert ...executable? without a negation" do
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

describe RuboCop::Cop::FormulaAudit::OptionDeclarations do
  subject(:cop) { described_class.new }

  it "build.without? in homebrew/core" do
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

  it "build.with? in homebrew/core" do
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

  it "build.without? in dependencies" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        depends_on "bar" if build.without?("baz")
                            ^^^^^^^^^^^^^^^^^^^^^ Use `:optional` or `:recommended` instead of `if build.without?("baz")`
      end
    RUBY
  end

  it "build.with? in dependencies" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        depends_on "bar" if build.with?("baz")
                            ^^^^^^^^^^^^^^^^^^ Use `:optional` or `:recommended` instead of `if build.with?("baz")`
      end
    RUBY
  end

  it "unless build.without? conditional" do
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

  it "unless build.with? conditional" do
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

  it "negated build.with? conditional" do
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

  it "negated build.without? conditional" do
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

  it "unnecessary build.without? conditional" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        desc "foo"
        url 'https://brew.sh/foo-1.0.tgz'
        def post_install
          return if build.without? "--without-bar"
                                    ^^^^^^^^^^^^^ Don't duplicate 'without': Use `build.without? \"bar\"` to check for \"--without-bar\"
        end
      end
    RUBY
  end

  it "unnecessary build.with? conditional" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        desc "foo"
        url 'https://brew.sh/foo-1.0.tgz'
        def post_install
          return if build.with? "--with-bar"
                                 ^^^^^^^^^^ Don't duplicate 'with': Use `build.with? \"bar\"` to check for \"--with-bar\"
        end
      end
    RUBY
  end

  it "build.include? deprecated" do
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

  it "def options usage" do
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

describe RuboCop::Cop::FormulaAudit::MpiCheck do
  subject(:cop) { described_class.new }

  context "When auditing formula" do
    it "reports an offense when using depends_on \"mpich\"" do
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

  context "When auditing popen commands" do
    it "Utils.popen_read should become Utils.safe_popen_read" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read "foo"
            ^^^^^^^^^^^^^^^^^^^^^^ Use `Utils.safe_popen_read` instead of `Utils.popen_read`
          end
        end
      RUBY
    end

    it "Utils.safe_popen_write should become Utils.popen_write" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_write "foo"
            ^^^^^^^^^^^^^^^^^^^^^^^ Use `Utils.safe_popen_write` instead of `Utils.popen_write`
          end
        end
      RUBY
    end

    it "does not correct Utils.popen_read in test block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install; end
          test do
            Utils.popen_read "foo"
          end
        end
      RUBY
    end

    it "corrects Utils.popen_read to Utils.safe_popen_read" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read "foo"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.safe_popen_read "foo"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "corrects Utils.popen_write to Utils.safe_popen_write" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_write "foo"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.safe_popen_write "foo"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "does not correct to Utils.safe_popen_read in test block" do
      source = <<~RUBY
        class Foo < Formula
          def install; end
          test do
            Utils.popen_write "foo"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(source)
    end
  end
end

describe RuboCop::Cop::FormulaAudit::ShellVariables do
  subject(:cop) { described_class.new }

  context "When auditing shell variables" do
    it "Shell variables should be expanded in Utils.popen" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen "SHELL=bash foo"
                         ^^^^^^^^^^^^^^ Use `Utils.popen({ "SHELL" => "bash" }, "foo")` instead of `Utils.popen "SHELL=bash foo"`
          end
        end
      RUBY
    end

    it "Shell variables should be expanded in Utils.safe_popen_read" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read "SHELL=bash foo"
                                   ^^^^^^^^^^^^^^ Use `Utils.safe_popen_read({ "SHELL" => "bash" }, "foo")` instead of `Utils.safe_popen_read "SHELL=bash foo"`
          end
        end
      RUBY
    end

    it "Shell variables should be expanded in Utils.safe_popen_write" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write "SHELL=bash foo"
                                    ^^^^^^^^^^^^^^ Use `Utils.safe_popen_write({ "SHELL" => "bash" }, "foo")` instead of `Utils.safe_popen_write "SHELL=bash foo"`
          end
        end
      RUBY
    end

    it "Shell variables should be expanded and keep inline string variables in the arguments" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen "SHELL=bash \#{bin}/foo"
                         ^^^^^^^^^^^^^^^^^^^^^ Use `Utils.popen({ "SHELL" => "bash" }, "\#{bin}/foo")` instead of `Utils.popen "SHELL=bash \#{bin}/foo"`
          end
        end
      RUBY
    end

    it "corrects shell variables in Utils.popen" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen("SHELL=bash foo")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen({ "SHELL" => "bash" }, "foo")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "corrects shell variables in Utils.safe_popen_read" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.safe_popen_read("SHELL=bash foo")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.safe_popen_read({ "SHELL" => "bash" }, "foo")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "corrects shell variables in Utils.safe_popen_write" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.safe_popen_write("SHELL=bash foo")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.safe_popen_write({ "SHELL" => "bash" }, "foo")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "corrects shell variables with inline string variable in arguments" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen("SHELL=bash \#{bin}/foo")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen({ "SHELL" => "bash" }, "\#{bin}/foo")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end
  end
end

describe RuboCop::Cop::FormulaAudit::LicenseArrays do
  subject(:cop) { described_class.new }

  context "When auditing licenses" do
    it "allow license strings" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license "MIT"
        end
      RUBY
    end

    it "allow license symbols" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license :public_domain
        end
      RUBY
    end

    it "allow license hashes" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", "0BSD"]
        end
      RUBY
    end

    it "require using :any_of instead of a license array" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license ["MIT", "0BSD"]
          ^^^^^^^^^^^^^^^^^^^^^^^ Use `license any_of: ["MIT", "0BSD"]` instead of `license ["MIT", "0BSD"]`
        end
      RUBY
    end

    it "corrects license arrays to hash with :any_of" do
      source = <<~RUBY
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license ["MIT", "0BSD"]
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", "0BSD"]
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end
  end
end

describe RuboCop::Cop::FormulaAudit::Licenses do
  subject(:cop) { described_class.new }

  context "When auditing licenses" do
    it "allow license strings" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license "MIT"
        end
      RUBY
    end

    it "allow license symbols" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license :public_domain
        end
      RUBY
    end

    it "allow license hashes" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license any_of: ["MIT", "0BSD"]
        end
      RUBY
    end

    it "allow license exceptions" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          license "MIT" => { with: "LLVM-exception" }
        end
      RUBY
    end

    it "allow multiline nested license hashes" do
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

    it "allow multiline nested license hashes with exceptions" do
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

    it "require multiple lines for nested license hashes" do
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

  context "When auditing python versions" do
    it "allow python with no dependency" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            puts "python@3.8"
          end
        end
      RUBY
    end

    it "allow non versioned python references" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python"
          end
        end
      RUBY
    end

    it "allow python with no version" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3"
          end
        end
      RUBY
    end

    it "allow matching versions" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.9"
          end
        end
      RUBY
    end

    it "allow matching versions without `@`" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.9"
          end
        end
      RUBY
    end

    it "allow matching versions with two digits" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.10"

          def install
            puts "python@3.10"
          end
        end
      RUBY
    end

    it "allow matching versions without `@` with two digits" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.10"

          def install
            puts "python3.10"
          end
        end
      RUBY
    end

    it "do not allow mismatching versions" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.8"
                 ^^^^^^^^^^^^ References to `python@3.8` should match the specified python dependency (`python@3.9`)
          end
        end
      RUBY
    end

    it "do not allow mismatching versions without `@`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.8"
                 ^^^^^^^^^^^ References to `python3.8` should match the specified python dependency (`python3.9`)
          end
        end
      RUBY
    end

    it "do not allow mismatching versions with two digits" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python@3.10"
                 ^^^^^^^^^^^^^ References to `python@3.10` should match the specified python dependency (`python@3.11`)
          end
        end
      RUBY
    end

    it "do not allow mismatching versions without `@` with two digits" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python3.10"
                 ^^^^^^^^^^^^ References to `python3.10` should match the specified python dependency (`python3.11`)
          end
        end
      RUBY
    end

    it "autocorrects mismatching versions" do
      source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.8"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python@3.9"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "autocorrects mismatching versions without `@`" do
      source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.8"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.9"

          def install
            puts "python3.9"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "autocorrects mismatching versions with two digits" do
      source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.10"

          def install
            puts "python@3.9"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.10"

          def install
            puts "python@3.10"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "autocorrects mismatching versions without `@` with two digits" do
      source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python3.10"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          depends_on "python@3.11"

          def install
            puts "python3.11"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end
  end
end

describe RuboCop::Cop::FormulaAudit::Miscellaneous do
  subject(:cop) { described_class.new }

  context "When auditing formula" do
    it "FileUtils usage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          FileUtils.mv "hello"
          ^^^^^^^^^^^^^^^^^^^^ Don\'t need \'FileUtils.\' before mv
        end
      RUBY
    end

    it "long inreplace block vars" do
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

    it "an invalid rebuild statement" do
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

    it "OS.linux? check" do
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

    it "fails_with :llvm block" do
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

    it "def test's deprecated usage" do
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

    it "with deprecated skip_clean call" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          skip_clean :all
          ^^^^^^^^^^^^^^^ `skip_clean :all` is deprecated; brew no longer strips symbols. Pass explicit paths to prevent Homebrew from removing empty folders.
        end
      RUBY
    end

    it "build.universal? deprecated usage" do
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

    it "build.universal? deprecation exempted formula" do
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

    it "deprecated ENV.universal_binary usage" do
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

    it "ENV.universal_binary deprecation exempted formula" do
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

    it "install_name_tool usage instead of ruby-macho" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "install_name_tool", "-id"
                  ^^^^^^^^^^^^^^^^^ Use ruby-macho instead of calling "install_name_tool"
        end
      RUBY
    end

    it "npm install without language::Node args" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "npm", "install"
          ^^^^^^^^^^^^^^^^^^^^^^^ Use Language::Node for npm install args
        end
      RUBY
    end

    it "npm install without language::Node args in kibana(exempted formula)" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/kibana@4.4.rb")
        class KibanaAT44 < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "npm", "install"
        end
      RUBY
    end

    it "depends_on with an instance as argument" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on FOO::BAR.new
                     ^^^^^^^^^^^^ `depends_on` can take requirement classes instead of instances
        end
      RUBY
    end

    it "non glob DIR usage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          rm_rf Dir["src/{llvm,test,librustdoc,etc/snapshot.pyc}"]
          rm_rf Dir["src/snapshot.pyc"]
                     ^^^^^^^^^^^^^^^^ Dir(["src/snapshot.pyc"]) is unnecessary; just use "src/snapshot.pyc"
        end
      RUBY
    end

    it "system call to fileUtils Method" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "mkdir", "foo"
                  ^^^^^ Use the `mkdir` Ruby method instead of `system "mkdir", "foo"`
        end
      RUBY
    end

    it "top-level function def outside class body" do
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

    it 'man+"man8" usage' do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            man1.install man+"man8" => "faad.1"
                              ^^^^ "man+"man8"" should be "man8"
          end
        end
      RUBY
    end

    it "hardcoded gcc compiler system" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "/usr/bin/gcc", "foo"
                    ^^^^^^^^^^^^ Use "#{ENV.cc}" instead of hard-coding "gcc"
          end
        end
      RUBY
    end

    it "hardcoded g++ compiler system" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "/usr/bin/g++", "-o", "foo", "foo.cc"
                    ^^^^^^^^^^^^ Use "#{ENV.cxx}" instead of hard-coding "g++"
          end
        end
      RUBY
    end

    it "hardcoded llvm-g++ compiler COMPILER_PATH" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            ENV["COMPILER_PATH"] = "/usr/bin/llvm-g++"
                                    ^^^^^^^^^^^^^^^^^ Use "#{ENV.cxx}" instead of hard-coding "llvm-g++"
          end
        end
      RUBY
    end

    it "hardcoded gcc compiler COMPILER_PATH" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            ENV["COMPILER_PATH"] = "/usr/bin/gcc"
                                    ^^^^^^^^^^^^ Use \"\#{ENV.cc}\" instead of hard-coding \"gcc\"
          end
        end
      RUBY
    end

    it "formula path shortcut : man" do
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

    it "formula path shortcut : libexec" do
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

    it "formula path shortcut : info" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "./configure", "--INFODIR=#{prefix}/share/info"
                                                       ^^^^^^ "#{prefix}/share" should be "#{share}"
                                                       ^^^^^^^^^^^ "#{prefix}/share/info" should be "#{info}"
          end
        end
      RUBY
    end

    it "formula path shortcut : man8" do
      expect_offense(<<~'RUBY')
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          def install
            system "./configure", "--MANDIR=#{prefix}/share/man/man8"
                                                      ^^^^^^ "#{prefix}/share" should be "#{share}"
                                                      ^^^^^^^^^^^^^^^ "#{prefix}/share/man/man8" should be "#{man8}"
          end
        end
      RUBY
    end

    it "dependencies which have to vendored" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "lpeg" => :lua51
                                ^^^^^ lua modules should be vendored rather than use deprecated `depends_on \"lpeg\" => :lua51`
        end
      RUBY
    end

    it "manually setting env" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          system "export", "var=value"
                  ^^^^^^ Use ENV instead of invoking 'export' to modify the environment
        end
      RUBY
    end

    it "dependencies with invalid options which lead to force rebuild" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "foo" => "with-bar"
                               ^^^^^^^^ Dependency foo should not use option with-bar
        end
      RUBY
    end

    it "dependencies with invalid options in array value which lead to force rebuild" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "httpd" => [:build, :test]
          depends_on "foo" => [:optional, "with-bar"]
                                           ^^^^^^^^ Dependency foo should not use option with-bar
          depends_on "icu4c" => [:optional, "c++11"]
                                             ^^^^^ Dependency icu4c should not use option c++11
        end
      RUBY
    end

    it "inspecting version manually" do
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

    it "deprecated ARGV.include? (--HEAD) usage" do
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

    it "deprecated needs :openmp usage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          needs :openmp
          ^^^^^^^^^^^^^ 'needs :openmp' should be replaced with 'depends_on \"gcc\"'
        end
      RUBY
    end

    it "deprecated MACOS_VERSION const usage" do
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

    it "deprecated if build.with? conditional dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on "foo" if build.with? "foo"
          ^^^^^^^^^^^^^^^^ Replace depends_on "foo" if build.with? "foo" with depends_on "foo" => :optional
        end
      RUBY
    end

    it "unless conditional dependency with build.without?" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'
          depends_on :foo unless build.without? "foo"
          ^^^^^^^^^^^^^^^ Replace depends_on :foo unless build.without? "foo" with depends_on :foo => :recommended
        end
      RUBY
    end

    it "unless conditional dependency with build.include?" do
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

  it "build-time checks in homebrew/core" do
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

  it "build-time checks in homebrew/core in allowlist" do
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

  context "When auditing shell commands" do
    it "system arguments should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            system "foo bar"
                   ^^^^^^^^^ Separate `system` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "system arguments with string interpolation should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            system "\#{bin}/foo bar"
                   ^^^^^^^^^^^^^^^^ Separate `system` commands into `\"\#{bin}/foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "system arguments with metacharacters should not be separated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            system "foo bar > baz"
          end
        end
      RUBY
    end

    it "only the first system argument should be separated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            system "foo", "bar baz"
          end
        end
      RUBY
    end

    it "Utils.popen arguments should not be separated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen("foo bar")
          end
        end
      RUBY
    end

    it "Utils.popen_read arguments should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo bar")
                             ^^^^^^^^^ Separate `Utils.popen_read` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "Utils.safe_popen_read arguments should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_read("foo bar")
                                  ^^^^^^^^^ Separate `Utils.safe_popen_read` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "Utils.popen_write arguments should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_write("foo bar")
                              ^^^^^^^^^ Separate `Utils.popen_write` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "Utils.safe_popen_write arguments should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.safe_popen_write("foo bar")
                                   ^^^^^^^^^ Separate `Utils.safe_popen_write` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "Utils.popen_read arguments with string interpolation should be separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("\#{bin}/foo bar")
                             ^^^^^^^^^^^^^^^^ Separate `Utils.popen_read` commands into `\"\#{bin}/foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "Utils.popen_read arguments with metacharacters should not be separated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo bar > baz")
          end
        end
      RUBY
    end

    it "only the first Utils.popen_read argument should be separated" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read("foo", "bar baz")
          end
        end
      RUBY
    end

    it "Utils.popen_read arguments should be separated following a shell variable" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            Utils.popen_read({ "SHELL" => "bash"}, "foo bar")
                                                   ^^^^^^^^^ Separate `Utils.popen_read` commands into `\"foo\", \"bar\"`
          end
        end
      RUBY
    end

    it "separates shell commands in system" do
      source = <<~RUBY
        class Foo < Formula
          def install
            system "foo bar"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            system "foo", "bar"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "separates shell commands with string interpolation in system" do
      source = <<~RUBY
        class Foo < Formula
          def install
            system "\#{foo}/bar baz"
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            system "\#{foo}/bar", "baz"
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "separates shell commands in Utils.popen_read" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read("foo bar")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read("foo", "bar")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "separates shell commands with string interpolation in Utils.popen_read" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read("\#{foo}/bar baz")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read("\#{foo}/bar", "baz")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "separates shell commands following a shell variable in Utils.popen_read" do
      source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read({ "SHELL" => "bash" }, "foo bar")
          end
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          def install
            Utils.popen_read({ "SHELL" => "bash" }, "foo", "bar")
          end
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end
  end
end
