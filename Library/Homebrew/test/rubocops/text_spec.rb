# typed: false
# frozen_string_literal: true

require "rubocops/text"

describe RuboCop::Cop::FormulaAudit::Text do
  subject(:cop) { described_class.new }

  context "when auditing formula text" do
    it 'reports an offense if `require "formula"` is present' do
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

    it "reports an offense if both openssl and libressl are dependencies" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          depends_on "openssl"
          depends_on "libressl" => :optional
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Formulae should not depend on both OpenSSL and LibreSSL (even optionally).
        end
      RUBY

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

    it "reports an offense if veclibfort is used instead of OpenBLAS (in homebrew/core)" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
          depends_on "veclibfort"
          ^^^^^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should use OpenBLAS as the default serial linear algebra library.
        end
      RUBY
    end

    it "reports an offense if lapack is used instead of OpenBLAS (in homebrew/core)" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
          depends_on "lapack"
          ^^^^^^^^^^^^^^^^^^^ Formulae in homebrew/core should use OpenBLAS as the default serial linear algebra library.
        end
      RUBY
    end

    it "reports an offense if xcodebuild is called without SYMROOT" do
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

    it "reports an offense if `go get` is executed" do
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

    it "reports an offense if `xcodebuild` is executed" do
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

    it "reports an offense if `plist_options` are not defined when using a formula-defined `plist`", :ruby23 do
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

    it 'reports an offense if `require "language/go"` is present' do
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

    it "reports an offense if formula uses virtualenv and also `setuptools` resource" do
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

    it "reports an offense if `Formula.factory(name)` is present" do
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

    it "reports an offense if `dep ensure` is used without `-vendor-only`" do
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

    it "reports an offense if `cargo build` is executed" do
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

    it "reports an offense if `make` calls are not separated" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            system "make && make install"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use separate `make` calls
          end
        end
      RUBY
    end

    it "reports an offense if paths are concatenated in string interpolation" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            ohai "foo \#{bar + "baz"}"
                      ^^^^^^^^^^^^^^ Do not concatenate paths in string interpolation
          end
        end
      RUBY
    end

    it 'reports an offense if `prefix + "bin"` is present' do
      expect_offense(<<~RUBY)
        class Foo < Formula
          def install
            ohai prefix + "bin"
                 ^^^^^^^^^^^^^^ Use `bin` instead of `prefix + "bin"`
          end
        end
      RUBY

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

  context "when auditing formula text in homebrew/core" do
    it "reports an offense if `env :userpaths` is present" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          env :userpaths
          ^^^^^^^^^^^^^^ `env :userpaths` in homebrew/core formulae is deprecated
        end
      RUBY
    end

    it "reports an offense if `env :std` is present in homebrew/core" do
      expect_offense(<<~RUBY, "/homebrew-core/")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          env :std
          ^^^^^^^^ `env :std` in homebrew/core formulae is deprecated
        end
      RUBY
    end

    it %Q(reports an offense if "\#{share}/<formula name>" is present) do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foo"
                 ^^^^^^^^^^^^^^ Use `\#{pkgshare}` instead of `\#{share}/foo`
          end
        end
      RUBY

      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foo/bar"
                 ^^^^^^^^^^^^^^^^^^ Use `\#{pkgshare}` instead of `\#{share}/foo`
          end
        end
      RUBY

      expect_offense(<<~RUBY, "/homebrew-core/Formula/foolibc++.rb")
        class Foolibcxx < Formula
          def install
            ohai "\#{share}/foolibc++"
                 ^^^^^^^^^^^^^^^^^^^^ Use `\#{pkgshare}` instead of `\#{share}/foolibc++`
          end
        end
      RUBY
    end

    it 'reports an offense if `share/"<formula name>"` is present' do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"foo"
                 ^^^^^^^^^^^ Use `pkgshare` instead of `share/"foo"`
          end
        end
      RUBY

      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"foo/bar"
                 ^^^^^^^^^^^^^^^ Use `pkgshare` instead of `share/"foo"`
          end
        end
      RUBY

      expect_offense(<<~RUBY, "/homebrew-core/Formula/foolibc++.rb")
        class Foolibcxx < Formula
          def install
            ohai share/"foolibc++"
                 ^^^^^^^^^^^^^^^^^ Use `pkgshare` instead of `share/"foolibc++"`
          end
        end
      RUBY
    end

    it %Q(reports no offenses if "\#{share}/<directory name>" doesn't match formula name) do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai "\#{share}/foo-bar"
          end
        end
      RUBY
    end

    it 'reports no offenses if `share/"<formula name>"` is not present' do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"foo-bar"
          end
        end
      RUBY

      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"bar"
          end
        end
      RUBY

      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          def install
            ohai share/"bar/foo"
          end
        end
      RUBY
    end

    it %Q(reports no offenses if formula name appears afer "\#{share}/<directory name>") do
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
