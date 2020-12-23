# typed: false
# frozen_string_literal: true

require "rubocops/components_order"

describe RuboCop::Cop::FormulaAudit::ComponentsOrder do
  subject(:cop) { described_class.new }

  context "When auditing formula components order" do
    it "When uses_from_macos precedes depends_on" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"
          url "https://brew.sh/foo-1.0.tgz"

          uses_from_macos "apple"
          depends_on "foo"
          ^^^^^^^^^^^^^^^^ `depends_on` (line 6) should be put before `uses_from_macos` (line 5)
        end
      RUBY
    end

    it "When license precedes sha256" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"
          url "https://brew.sh/foo-1.0.tgz"
          license "0BSD"
          sha256 "samplesha256"
          ^^^^^^^^^^^^^^^^^^^^^ `sha256` (line 5) should be put before `license` (line 4)
        end
      RUBY
    end

    it "When `bottle` precedes `livecheck`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"
          url "https://brew.sh/foo-1.0.tgz"

          bottle :unneeded

          livecheck do
          ^^^^^^^^^^^^ `livecheck` (line 7) should be put before `bottle` (line 5)
            url "https://brew.sh/foo/versions/"
            regex(/href=.+?foo-(\d+(?:\.\d+)+)\.t/)
          end
        end
      RUBY
    end

    it "When url precedes homepage" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
          ^^^^^^^^^^^^^^^^^^^^^^^^^^ `homepage` (line 3) should be put before `url` (line 2)
        end
      RUBY
    end

    it "When `resource` precedes `depends_on`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource "foo2" do
            url "https://brew.sh/foo-2.0.tgz"
          end

          depends_on "openssl"
          ^^^^^^^^^^^^^^^^^^^^ `depends_on` (line 8) should be put before `resource` (line 4)
        end
      RUBY
    end

    it "When `test` precedes `plist`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          test do
            expect(shell_output("./dogs")).to match("Dogs are terrific")
          end

          def plist
          ^^^^^^^^^ `plist` (line 8) should be put before `test` (line 4)
          end
        end
      RUBY
    end

    it "When `install` precedes `depends_on`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          def install
          end

          depends_on "openssl"
          ^^^^^^^^^^^^^^^^^^^^ `depends_on` (line 7) should be put before `install` (line 4)
        end
      RUBY
    end

    it "When `test` precedes `depends_on`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          def install
          end

          def test
          end

          depends_on "openssl"
          ^^^^^^^^^^^^^^^^^^^^ `depends_on` (line 10) should be put before `install` (line 4)
        end
      RUBY
    end

    it "When only one of many `depends_on` precedes `conflicts_with`" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          depends_on "autoconf" => :build
          conflicts_with "visionmedia-watch"
          depends_on "automake" => :build
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ `depends_on` (line 4) should be put before `conflicts_with` (line 3)
          depends_on "libtool" => :build
          depends_on "pkg-config" => :build
          depends_on "gettext"
        end
      RUBY
    end

    it "the on_macos block is not after uses_from_macos" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          on_macos do
            depends_on "readline"
          end
          uses_from_macos "bar"
          ^^^^^^^^^^^^^^^^^^^^^ `uses_from_macos` (line 6) should be put before `on_macos` (line 3)
        end
      RUBY
    end

    it "the on_linux block is not after uses_from_macos" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          on_linux do
            depends_on "readline"
          end
          uses_from_macos "bar"
          ^^^^^^^^^^^^^^^^^^^^^ `uses_from_macos` (line 6) should be put before `on_linux` (line 3)
        end
      RUBY
    end

    it "the on_linux block is not after the on_macos block" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          on_linux do
            depends_on "vim"
          end
          on_macos do
          ^^^^^^^^^^^ `on_macos` (line 6) should be put before `on_linux` (line 3)
            depends_on "readline"
          end
        end
      RUBY
    end
  end

  context "When auditing formula components order with autocorrect" do
    it "When url precedes homepage" do
      source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"
        end
      RUBY

      correct_source = <<~RUBY
        class Foo < Formula
          homepage "https://brew.sh"
          url "https://brew.sh/foo-1.0.tgz"
        end
      RUBY

      corrected_source = autocorrect_source(source)
      expect(corrected_source).to eq(correct_source)
    end

    it "When `resource` precedes `depends_on`" do
      source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource "foo2" do
            url "https://brew.sh/foo-2.0.tgz"
          end

          depends_on "openssl"
        end
      RUBY

      correct_source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          depends_on "openssl"

          resource "foo2" do
            url "https://brew.sh/foo-2.0.tgz"
          end
        end
      RUBY

      corrected_source = autocorrect_source(source)
      expect(corrected_source).to eq(correct_source)
    end

    it "When `depends_on` precedes `deprecate!`" do
      source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          depends_on "openssl"

          deprecate! because: "has been replaced by bar"
        end
      RUBY

      correct_source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          deprecate! because: "has been replaced by bar"

          depends_on "openssl"
        end
      RUBY

      corrected_source = autocorrect_source(source)
      expect(corrected_source).to eq(correct_source)
    end
  end

  context "no on_os_block" do
    it "does not fail when there is no on_os block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"

          depends_on "pkg-config" => :build

          def install
          end
        end
      RUBY
    end
  end

  context "on_os_block" do
    it "correctly uses on_macos and on_linux blocks" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"

          depends_on "pkg-config" => :build

          uses_from_macos "libxml2"

          on_macos do
            depends_on "perl"

            resource "resource1" do
              url "https://brew.sh/resource1.tar.gz"
              sha256 "a2f5650770e1c87fb335af19a9b7eb73fc05ccf22144eb68db7d00cd2bcb0902"

              patch do
                url "https://raw.githubusercontent.com/Homebrew/formula-patches/0ae366e6/patch3.diff"
                sha256 "89fa3c95c329ec326e2e76493471a7a974c673792725059ef121e6f9efb05bf4"
              end
            end

            resource "resource2" do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            patch do
              url "https://raw.githubusercontent.com/Homebrew/formula-patches/0ae366e6/patch1.diff"
              sha256 "89fa3c95c329ec326e2e76493471a7a974c673792725059ef121e6f9efb05bf4"
            end

            patch do
              url "https://raw.githubusercontent.com/Homebrew/formula-patches/0ae366e6/patch2.diff"
              sha256 "89fa3c95c329ec326e2e76493471a7a974c673792725059ef121e6f9efb05bf4"
            end
          end

          on_linux do
            depends_on "readline"
          end

          def install
          end
        end
      RUBY
    end
  end

  context "on_macos_block" do
    it "correctly uses as single on_macos block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"

          on_macos do
            disable! because: :does_not_build
            depends_on "readline"
          end

          def install
          end
        end
      RUBY
    end
  end

  context "on_linux_block" do
    it "correctly uses as single on_linux block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"

          on_linux do
            deprecate! because: "it's deprecated"
            depends_on "readline"
          end

          def install
          end
        end
      RUBY
    end
  end

  it "there can only be one on_macos block" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        on_macos do
          depends_on "readline"
        end

        on_macos do
        ^^^^^^^^^^^ there can only be one `on_macos` block in a formula.
          depends_on "foo"
        end
      end
    RUBY
  end

  it "there can only be one on_linux block" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        on_linux do
          depends_on "readline"
        end

        on_linux do
        ^^^^^^^^^^^ there can only be one `on_linux` block in a formula.
          depends_on "foo"
        end
      end
    RUBY
  end

  it "the on_macos block can only contain depends_on, patch and resource nodes" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        on_macos do
          depends_on "readline"
          uses_from_macos "ncurses"
          ^^^^^^^^^^^^^^^^^^^^^^^^^ `on_macos` cannot include `uses_from_macos`. [...]
        end
      end
    RUBY
  end

  it "the on_linux block can only contain depends_on, patch and resource nodes" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        on_linux do
          depends_on "readline"
          uses_from_macos "ncurses"
          ^^^^^^^^^^^^^^^^^^^^^^^^^ `on_linux` cannot include `uses_from_macos`. [...]
        end
      end
    RUBY
  end

  context "resource" do
    it "correctly uses an on_macos and on_linux block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"

          resource do
            on_macos do
              url "https://brew.sh/resource1.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            on_linux do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "correctly uses an on_macos and on_linux block with versions" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          homepage "https://brew.sh"

          resource do
            on_macos do
              url "https://brew.sh/resource1.tar.gz"
              version "1.2.3"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            on_linux do
              url "https://brew.sh/resource2.tar.gz"
              version "1.2.3"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "there are two on_macos blocks, which is not allowed" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource do
          ^^^^^^^^^^^ there can only be one `on_macos` block in a resource block.
            on_macos do
              url "https://brew.sh/resource1.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            on_macos do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "there are two on_linux blocks, which is not allowed" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource do
          ^^^^^^^^^^^ there can only be one `on_linux` block in a resource block.
            on_linux do
              url "https://brew.sh/resource1.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            on_linux do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "there is a on_macos block but no on_linux block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          resource do
            on_macos do
              url "https://brew.sh/resource1.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "there is a on_linux block but no on_macos block" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          resource do
            on_linux do
              url "https://brew.sh/resource1.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "the content of the on_macos block is wrong in a resource block" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource do
          ^^^^^^^^^^^ `on_macos` blocks within resource blocks must contain only a url and sha256 or a url, version, and sha256 (in those orders).
            on_macos do
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
              url "https://brew.sh/resource2.tar.gz"
            end

            on_linux do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "the content of the on_macos block is wrong and not a method" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource do
          ^^^^^^^^^^^ `on_macos` blocks within resource blocks must contain only a url and sha256 or a url, version, and sha256 (in those orders).
            on_macos do
              if foo == :bar
                url "https://brew.sh/resource2.tar.gz"
                sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
              else
                url "https://brew.sh/resource1.tar.gz"
                sha256 "686372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
              end
            end

            on_linux do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end
          end
        end
      RUBY
    end

    it "the content of the on_linux block is wrong in a resource block" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource do
          ^^^^^^^^^^^ `on_linux` blocks within resource blocks must contain only a url and sha256 or a url, version, and sha256 (in those orders).
            on_macos do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            on_linux do
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
              url "https://brew.sh/resource2.tar.gz"
            end
          end
        end
      RUBY
    end

    it "the content of the on_linux block is wrong and not a method" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          resource do
          ^^^^^^^^^^^ `on_linux` blocks within resource blocks must contain only a url and sha256 or a url, version, and sha256 (in those orders).
            on_macos do
              url "https://brew.sh/resource2.tar.gz"
              sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
            end

            on_linux do
              if foo == :bar
                url "https://brew.sh/resource2.tar.gz"
                sha256 "586372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
              else
                url "https://brew.sh/resource1.tar.gz"
                sha256 "686372eb92059873e29eba4f9dec8381541b4d3834660707faf8ba59146dfc35"
              end
            end
          end
        end
      RUBY
    end
  end
end
