# typed: false
# frozen_string_literal: true

require "rubocops/lines"

describe RuboCop::Cop::FormulaAudit::OSConditionals do
  subject(:cop) { described_class.new }

  context "when auditing OS conditionals" do
    it "reports an offense when `OS.linux?` is used on Formula class" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          desc "foo"
          if OS.linux?
             ^^^^^^^^^ Don't use 'if OS.linux?', use 'on_linux do' instead.
            url 'https://brew.sh/linux-1.0.tgz'
          else
            url 'https://brew.sh/linux-1.0.tgz'
          end
        end
      RUBY
    end

    it "reports an offense when `OS.mac?` is used on Formula class" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          desc "foo"
          if OS.mac?
             ^^^^^^^ Don't use 'if OS.mac?', use 'on_macos do' instead.
            url 'https://brew.sh/mac-1.0.tgz'
          else
            url 'https://brew.sh/linux-1.0.tgz'
          end
        end
      RUBY
    end

    it "reports an offense when `on_macos` is used in install method" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'

          def install
            on_macos do
            ^^^^^^^^ Don't use 'on_macos' in 'def install', use 'if OS.mac?' instead.
              true
            end
          end
        end
      RUBY
    end

    it "reports an offense when `on_linux` is used in install method" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'

          def install
            on_linux do
            ^^^^^^^^ Don't use 'on_linux' in 'def install', use 'if OS.linux?' instead.
              true
            end
          end
        end
      RUBY
    end

    it "reports an offense when `on_macos` is used in test block" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          desc "foo"
          url 'https://brew.sh/foo-1.0.tgz'

          test do
            on_macos do
            ^^^^^^^^ Don't use 'on_macos' in 'test do', use 'if OS.mac?' instead.
              true
            end
          end
        end
      RUBY
    end
  end
end
