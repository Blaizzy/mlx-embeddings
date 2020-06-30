# frozen_string_literal: true

require "rubocops/deprecate"

describe RuboCop::Cop::FormulaAudit::Deprecate do
  subject(:cop) { described_class.new }

  context "When auditing formula for deprecate!" do
    it "deprecation date is not ISO 8601 compliant" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          deprecate! :date => "June 25, 2020"
                              ^^^^^^^^^^^^^^^ Use `2020-06-25` to comply with ISO 8601
        end
      RUBY
    end

    it "deprecation date is ISO 8601 compliant" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          deprecate! :date => "2020-06-25"
        end
      RUBY
    end

    it "no deprecation date" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          deprecate!
        end
      RUBY
    end

    it "auto corrects to ISO 8601 format" do
      source = <<~RUBY
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          deprecate! :date => "June 25, 2020"
        end
      RUBY

      corrected_source = <<~RUBY
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          deprecate! :date => "2020-06-25"
        end
      RUBY

      new_source = autocorrect_source(source)
      expect(new_source).to eq(corrected_source)
    end
  end
end
