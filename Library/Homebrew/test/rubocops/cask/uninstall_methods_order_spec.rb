# frozen_string_literal: true

require "rubocops/rubocop-cask"
describe RuboCop::Cop::Cask::UninstallMethodsOrder, :config do
  context "with order errors in both the uninstall and zap block" do
    it "reports an offense and corrects the order" do
      expect_offense(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          uninstall delete:  [
                    ^^^^^^ `delete` method out of order
                      "/usr/local/bin/foo",
                      "/usr/local/bin/foobar",
                    ],
                    script:  {
                    ^^^^^^ `script` method out of order
                      executable: "/usr/local/bin/foo",
                      sudo:       false,
                    },
                    pkgutil: "org.foo.bar"
                    ^^^^^^^ `pkgutil` method out of order

          zap delete: [
                "~/Library/Application Support/Bar",
                "~/Library/Application Support/Foo",
              ],
              rmdir:  "~/Library/Application Support",
              ^^^^^ `rmdir` method out of order
              trash:  "~/Library/Application Support/FooBar"
              ^^^^^ `trash` method out of order
        end
      CASK

      expect_correction(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          uninstall script:  {
                      executable: "/usr/local/bin/foo",
                      sudo:       false,
                    },
                    pkgutil: "org.foo.bar",
                    delete:  [
                      "/usr/local/bin/foo",
                      "/usr/local/bin/foobar",
                    ]

          zap delete: [
                "~/Library/Application Support/Bar",
                "~/Library/Application Support/Foo",
              ],
              trash:  "~/Library/Application Support/FooBar",
              rmdir:  "~/Library/Application Support"
        end
      CASK
    end
  end

  context "with incorrectly ordered uninstall methods" do
    it "reports an offense and corrects the order" do
      expect_offense(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

        uninstall delete:  [
                  ^^^^^^ `delete` method out of order
                    "/usr/local/bin/foo",
                    "/usr/local/bin/foobar",
                  ],
                  script:  {
                  ^^^^^^ `script` method out of order
                    executable: "/usr/local/bin/foo",
                    sudo:       false,
                  },
                  pkgutil: "org.foo.bar"
                  ^^^^^^^ `pkgutil` method out of order
        end
      CASK

      expect_correction(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

        uninstall script:  {
                    executable: "/usr/local/bin/foo",
                    sudo:       false,
                  },
                  pkgutil: "org.foo.bar",
                  delete:  [
                    "/usr/local/bin/foo",
                    "/usr/local/bin/foobar",
                  ]
        end
      CASK
    end
  end

  context "with correctly ordered uninstall methods" do
    it "does not report an offense" do
      expect_no_offenses(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          uninstall script:  {
                      executable: "/usr/local/bin/foo",
                      sudo:       false,
                    },
                    pkgutil: "org.foo.bar",
                    delete:  [
                      "/usr/local/bin/foo",
                      "/usr/local/bin/foobar",
                    ]
        end
      CASK
    end
  end

  context "with a single method in uninstall block" do
    it "does not report an offense" do
      expect_no_offenses(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          uninstall delete: "/usr/local/bin/foo"
        end
      CASK
      expect_no_offenses(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          uninstall pkgutil: [
            "org.foo.bar",
            "org.foobar.bar",
          ]
        end
      CASK
    end
  end

  context "with incorrectly ordered zap methods" do
    it "reports an offense and corrects the order" do
      expect_offense(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          zap delete: [
                "~/Library/Application Support/Foo",
                "~/Library/Application Support/Bar",
              ],
              rmdir: "~/Library/Application Support",
              ^^^^^ `rmdir` method out of order
              trash: "~/Library/Application Support/FooBar"
              ^^^^^ `trash` method out of order
        end
      CASK

      expect_correction(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          zap delete: [
                "~/Library/Application Support/Foo",
                "~/Library/Application Support/Bar",
              ],
              trash: "~/Library/Application Support/FooBar",
              rmdir: "~/Library/Application Support"
        end
      CASK
    end
  end

  context "with correctly ordered zap methods" do
    it "does not report an offense" do
      expect_no_offenses(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          zap delete: [
                "~/Library/Application Support/Bar",
                "~/Library/Application Support/Foo",
              ],
              trash:  "~/Library/Application Support/FooBar",
              rmdir:  "~/Library/Application Support"
        end
      CASK
    end
  end

  context "with a single method in the zap block" do
    it "does not report an offense" do
      expect_no_offenses(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          zap trash:  "~/Library/Application Support/FooBar"
        end
      CASK
      expect_no_offenses(<<~CASK)
        cask "foo" do
          url "https://example.com/foo.zip"

          zap trash: [
            "~/Library/Application Support/FooBar",
            "~/Library/Application Support/FooBarBar",
          ]
        end
      CASK
    end
  end
end
