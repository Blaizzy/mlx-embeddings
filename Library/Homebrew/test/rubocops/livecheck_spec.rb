# typed: false
# frozen_string_literal: true

require "rubocops/livecheck"

describe RuboCop::Cop::FormulaAudit::LivecheckSkip do
  subject(:cop) { described_class.new }

  it "reports an offense when a skipped formula's livecheck block contains other information" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
        ^^^^^^^^^^^^ Skipped formulae must not contain other livecheck information.
          skip "Not maintained"
          url :stable
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          skip "Not maintained"
        end
      end
    RUBY
  end

  it "reports no offenses when a skipped formula's livecheck block contains no other information" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          skip "Not maintained"
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAudit::LivecheckUrlProvided do
  subject(:cop) { described_class.new }

  it "reports an offense when a `url` is not specified in the livecheck block" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
        ^^^^^^^^^^^^ A `url` must be provided to livecheck.
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end

  it "reports no offenses when a `url` is specified in the livecheck block" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAudit::LivecheckUrlSymbol do
  subject(:cop) { described_class.new }

  it "reports an offense when the `url` specified in the livecheck block is identical to a formula URL" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url "https://brew.sh/foo-1.0.tgz"
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use `url :stable`
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
        end
      end
    RUBY
  end

  it "reports no offenses when the `url` specified in the livecheck block is not identical to a formula URL" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url "https://brew.sh/foo/releases/"
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAudit::LivecheckRegexParentheses do
  subject(:cop) { described_class.new }

  it "reports an offense when the `regex` call in the livecheck block does not use parentheses" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex %r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ The `regex` call should always use parentheses.
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end

  it "reports no offenses when the `regex` call in the livecheck block uses parentheses" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAudit::LivecheckRegexExtension do
  subject(:cop) { described_class.new }

  it "reports an offense when the `regex` does not use `\\.t` for archive file extensions" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.tgz}i)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Use `\\.t` instead of `\\.tgz`
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end

  it "reports no offenses when the `regex` uses `\\.t` for archive file extensions" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAudit::LivecheckRegexIfPageMatch do
  subject(:cop) { described_class.new }

  it "reports an offense when there is no `regex` for `strategy :page_match`" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
        ^^^^^^^^^^^^ A `regex` is required if `strategy :page_match` is present.
          url :stable
          strategy :page_match
        end
      end
    RUBY
  end

  it "rreports no offenses when a `regex` is specified for `strategy :page_match`" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          strategy :page_match
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end
end

describe RuboCop::Cop::FormulaAudit::LivecheckRegexCaseInsensitive do
  subject(:cop) { described_class.new }

  it "reports an offense when the `regex` is not case-insensitive" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t})
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Regexes should be case-insensitive unless sensitivity is explicitly required for proper matching.
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end

  it "reports no offenses when the `regex` is case-insensitive" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        livecheck do
          url :stable
          regex(%r{href=.*?/formula[._-]v?(\\d+(?:\\.\\d+)+)\\.t}i)
        end
      end
    RUBY
  end
end
