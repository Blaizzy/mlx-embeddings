# typed: false
# frozen_string_literal: true

require "rubocops/lines"

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
