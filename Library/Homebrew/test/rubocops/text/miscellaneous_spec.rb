# typed: false
# frozen_string_literal: true

require "rubocops/lines"

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
