# frozen_string_literal: true

require "rubocops/conflicts"

describe RuboCop::Cop::FormulaAudit::Conflicts do
  subject(:cop) { described_class.new }

  context "When auditing conflicts_with" do
    it "conflicts_with reason is capitalized" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo.rb")
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          conflicts_with "bar", :because => "Reason"
                                            ^^^^^^^^ 'Reason' from the `conflicts_with` reason should be 'reason'.
          conflicts_with "baz", :because => "Foo is the formula name which does not require downcasing"
        end
      RUBY
    end

    it "conflicts_with reason ends with a period" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          conflicts_with "bar", "baz", :because => "reason."
                                                   ^^^^^^^^^ `conflicts_with` reason should not end with a period.
        end
      RUBY
    end

    it "conflicts_with in a versioned formula" do
      expect_offense(<<~RUBY, "/homebrew-core/Formula/foo@2.0.rb")
        class FooAT20 < Formula
          url 'https://brew.sh/foo-2.0.tgz'
          conflicts_with "mysql", "mariadb"
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Versioned formulae should not use `conflicts_with`. Use `keg_only :versioned_formula` instead.
        end
      RUBY
    end

    it "no conflicts_with" do
      expect_no_offenses(<<~RUBY, "/homebrew-core/Formula/foo@2.0.rb")
        class FooAT20 < Formula
          url 'https://brew.sh/foo-2.0.tgz'
          homepage "https://brew.sh"
        end
      RUBY
    end

    it "auto-corrects capitalized reason" do
      source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          conflicts_with "bar", :because => "Reason"
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          conflicts_with "bar", :because => "reason"
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end

    it "auto-corrects trailing period" do
      source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          conflicts_with "bar", :because => "reason."
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          conflicts_with "bar", :because => "reason"
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end
  end

  include_examples "formulae exist", described_class::ALLOWLIST
end
