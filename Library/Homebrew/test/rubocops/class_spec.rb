# typed: false
# frozen_string_literal: true

require "rubocops/class"

describe RuboCop::Cop::FormulaAudit::ClassName do
  subject(:cop) { described_class.new }

  corrected_source = <<~RUBY
    class Foo < Formula
      url 'https://brew.sh/foo-1.0.tgz'
    end
  RUBY

  it "reports and corrects an offense when using ScriptFileFormula" do
    expect_offense(<<~RUBY)
      class Foo < ScriptFileFormula
                  ^^^^^^^^^^^^^^^^^ ScriptFileFormula is deprecated, use Formula instead
        url 'https://brew.sh/foo-1.0.tgz'
      end
    RUBY
    expect_correction(corrected_source)
  end

  it "reports and corrects an offense when using GithubGistFormula" do
    expect_offense(<<~RUBY)
      class Foo < GithubGistFormula
                  ^^^^^^^^^^^^^^^^^ GithubGistFormula is deprecated, use Formula instead
        url 'https://brew.sh/foo-1.0.tgz'
      end
    RUBY
    expect_correction(corrected_source)
  end

  it "reports and corrects an offense when using AmazonWebServicesFormula" do
    expect_offense(<<~RUBY)
      class Foo < AmazonWebServicesFormula
                  ^^^^^^^^^^^^^^^^^^^^^^^^ AmazonWebServicesFormula is deprecated, use Formula instead
        url 'https://brew.sh/foo-1.0.tgz'
      end
    RUBY
    expect_correction(corrected_source)
  end
end

describe RuboCop::Cop::FormulaAudit::Test do
  subject(:cop) { described_class.new }

  it "reports and corrects an offense when /usr/local/bin is found in test calls" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url 'https://brew.sh/foo-1.0.tgz'

        test do
          system "/usr/local/bin/test"
                 ^^^^^^^^^^^^^^^^^^^^^ use \#{bin} instead of /usr/local/bin in system
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url 'https://brew.sh/foo-1.0.tgz'

        test do
          system "\#{bin}/test"
        end
      end
    RUBY
  end

  it "reports and corrects an offense when passing 0 as the second parameter to shell_output" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url 'https://brew.sh/foo-1.0.tgz'

        test do
          shell_output("\#{bin}/test", 0)
                                      ^ Passing 0 to shell_output() is redundant
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url 'https://brew.sh/foo-1.0.tgz'

        test do
          shell_output("\#{bin}/test")
        end
      end
    RUBY
  end

  it "reports an offense when there is an empty test block" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url 'https://brew.sh/foo-1.0.tgz'

        test do
        ^^^^^^^ `test do` should not be empty
        end
      end
    RUBY
  end

  it "reports an offense when test is falsely true" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url 'https://brew.sh/foo-1.0.tgz'

        test do
        ^^^^^^^ `test do` should contain a real test
          true
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAuditStrict::TestPresent do
  subject(:cop) { described_class.new }

  it "reports an offense when there is no test block" do
    expect_offense(<<~RUBY)
      class Foo < Formula
      ^^^^^^^^^^^^^^^^^^^ A `test do` test block should be added
        url 'https://brew.sh/foo-1.0.tgz'
      end
    RUBY
  end
end
