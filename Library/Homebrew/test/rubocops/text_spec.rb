# typed: false
# frozen_string_literal: true

require "rubocops/text"

describe RuboCop::Cop::FormulaAudit::Text do
  subject(:cop) { described_class.new }

  context "When auditing formula text" do
    it "with `require \"formula\"` is present" do
      expect_offense(<<~RUBY)
        require "formula"
        ^^^^^^^^^^^^^^^^^ `require "formula"` is now unnecessary
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
        end
      RUBY

      expect_correction(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
        end
      RUBY
    end

    it "with both openssl and libressl optional dependencies" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          depends_on "openssl"
          depends_on "libressl" => :optional
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Formulae should not depend on both OpenSSL and LibreSSL (even optionally).
        end
      RUBY
    end

    it "with both openssl and libressl dependencies" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          depends_on "openssl"
          depends_on "libressl"
          ^^^^^^^^^^^^^^^^^^^^^ Formulae should not depend on both OpenSSL and LibreSSL (even optionally).
        end
      RUBY
    end

    it "when veclibfort is used instead of OpenBLAS" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
          depends_on "veclibfort"
          ^^^^^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should use OpenBLAS as the default serial linear algebra library.
        end
      RUBY
    end

    it "when lapack is used instead of OpenBLAS" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
          depends_on "lapack"
          ^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should use OpenBLAS as the default serial linear algebra library.
        end
      RUBY
    end

    it "When xcodebuild is called without SYMROOT" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            xcodebuild "-project", "meow.xcodeproject"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ xcodebuild should be passed an explicit \"SYMROOT\"
          end
        end
      RUBY
    end

    it "When xcodebuild is called without any args" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            xcodebuild
            ^^^^^^^^^^ xcodebuild should be passed an explicit \"SYMROOT\"
          end
        end
      RUBY
    end

    it "When go get is executed" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            system "go", "get", "bar"
            ^^^^^^^^^^^^^^^^^^^^^^^^^ Do not use `go get`. Please ask upstream to implement Go vendoring
          end
        end
      RUBY
    end

    it "When xcodebuild is executed" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            system "xcodebuild", "foo", "bar"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ use \"xcodebuild *args\" instead of \"system 'xcodebuild', *args\"
          end
        end
      RUBY
    end

    it "When plist_options are not defined when using a formula-defined plist", :ruby23 do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            system "xcodebuild", "foo", "bar"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ use \"xcodebuild *args\" instead of \"system 'xcodebuild', *args\"
          end

          def plist
          ^^^^^^^^^ Please set plist_options when using a formula-defined plist.
            <<~XML
              <?xml version="1.0" encoding="UTF-8"?>
              <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
              <plist version="1.0">
              <dict>
                <key>Label</key>
                <string>org.nrpe.agent</string>
              </dict>
              </plist>
            XML
          end
        end
      RUBY
    end

    it "When language/go is require'd" do
      expect_offense(<<~RUBY)
        require "language/go"
        ^^^^^^^^^^^^^^^^^^^^^ require "language/go" is unnecessary unless using `go_resource`s

        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            system "go", "get", "bar"
            ^^^^^^^^^^^^^^^^^^^^^^^^^ Do not use `go get`. Please ask upstream to implement Go vendoring
          end
        end
      RUBY
    end

    it "When formula uses virtualenv and also `setuptools` resource" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          resource "setuptools" do
          ^^^^^^^^^^^^^^^^^^^^^ Formulae using virtualenvs do not need a `setuptools` resource.
            url "https://foo.com/foo.tar.gz"
            sha256 "db0904a28253cfe53e7dedc765c71596f3c53bb8a866ae50123320ec1a7b73fd"
          end

          def install
            virtualenv_create(libexec)
          end
        end
      RUBY
    end

    it "When Formula.factory(name) is used" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            Formula.factory(name)
            ^^^^^^^^^^^^^^^^^^^^^ \"Formula.factory(name)\" is deprecated in favor of \"Formula[name]\"
          end
        end
      RUBY
    end

    it "When dep ensure is used without `-vendor-only`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            system "dep", "ensure"
            ^^^^^^^^^^^^^^^^^^^^^^ use \"dep\", \"ensure\", \"-vendor-only\"
          end
        end
      RUBY
    end

    it "When cargo build is executed" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          def install
            system "cargo", "build"
            ^^^^^^^^^^^^^^^^^^^^^^^ use \"cargo\", \"install\", *std_cargo_args
          end
        end
      RUBY
    end

    it "When make calls are not separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            system "make && make install"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use separate `make` calls
          end
        end
      RUBY
    end

    it "When concatenating in string interpolation" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            ohai "foo \#{bar + "baz"}"
                      ^^^^^^^^^^^^^^ Do not concatenate paths in string interpolation
          end
        end
      RUBY
    end

    it "When using `prefix + \"bin\"` instead of `bin`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            ohai prefix + "bin"
                 ^^^^^^^^^^^^^^ Use `bin` instead of `prefix + "bin"`
          end
        end
      RUBY
    end

    it "When using `prefix + \"bin/foo\"` instead of `bin`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            ohai prefix + "bin/foo"
                 ^^^^^^^^^^^^^^^^^^ Use `bin` instead of `prefix + "bin"`
          end
        end
      RUBY
    end
  end
end

describe RuboCop::Cop::FormulaAuditStrict::Text do
  subject(:cop) { described_class.new }

  context "When auditing formula text" do
    it "when deprecated `env :userpaths` is present" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          env :userpaths
          ^^^^^^^^^^^^^^ `env :userpaths` in homebrew/core formulae is deprecated
        end
      RUBY
    end

    it "when deprecated `env :std` is present in homebrew-core" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          env :std
          ^^^^^^^^ `env :std` in homebrew/core formulae is deprecated
        end
      RUBY
    end

    it "when `\#{share}/foo` is used instead of `\#{pkgshare}`" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foo"
                 ^^^^^^^^^^^^^^ Use `\#{pkgshare}` instead of `\#{share}/foo`
          end
        end
      RUBY
    end

    it "when `\#{share}/foo/bar` is used instead of `\#{pkgshare}/bar`" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foo/bar"
                 ^^^^^^^^^^^^^^^^^^ Use `\#{pkgshare}` instead of `\#{share}/foo`
          end
        end
      RUBY
    end

    it "when `\#{share}/foolibc++` is used instead of `\#{pkgshare}/foolibc++`" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foolibc++.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foolibc++"
                 ^^^^^^^^^^^^^^^^^^^^ Use `\#{pkgshare}` instead of `\#{share}/foolibc++`
          end
        end
      RUBY
    end

    it "when `share/\"foo\"` is used instead of `pkgshare`" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"foo"
                 ^^^^^^^^^^^ Use `pkgshare` instead of `share/"foo"`
          end
        end
      RUBY
    end

    it "when `share/\"foo/bar\"` is used instead of `pkgshare`" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"foo/bar"
                 ^^^^^^^^^^^^^^^ Use `pkgshare` instead of `share/"foo"`
          end
        end
      RUBY
    end

    it "when `share/\"foolibc++\"` is used instead of `pkgshare`" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foolibc++.rb")
        class Foo < Formula
          def install
            ohai share/"foolibc++"
                 ^^^^^^^^^^^^^^^^^ Use `pkgshare` instead of `share/"foolibc++"`
          end
        end
      RUBY
    end

    it "when `\#{share}/foo-bar` doesn't match formula name" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foo-bar"
          end
        end
      RUBY
    end

    it "when `share/foo-bar` doesn't match formula name" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"foo-bar"
          end
        end
      RUBY
    end

    it "when `share/bar` doesn't match formula name" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"bar"
          end
        end
      RUBY
    end

    it "when formula name appears afer `share/\"bar\"`" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"bar/foo"
          end
        end
      RUBY
    end

    it "when formula name appears afer `\"\#{share}/bar\"`" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/bar/foo"
          end
        end
      RUBY
    end
  end
end
