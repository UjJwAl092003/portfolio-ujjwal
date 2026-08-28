

(function ($) {
  var $window = $(window),
    $head = $("head"),
    $body = $("body");

  // Breakpoints.
  breakpoints({
    xlarge: ["1281px", "1680px"],
    large: ["981px", "1280px"],
    medium: ["737px", "980px"],
    small: ["481px", "736px"],
    xsmall: ["361px", "480px"],
    xxsmall: [null, "360px"],
    "xlarge-to-max": "(min-width: 1681px)",
    "small-to-xlarge": "(min-width: 481px) and (max-width: 1680px)",
  });

  // Stops animations/transitions until the page has ...

  // ... loaded.
  $window.on("load", function () {
    window.setTimeout(function () {
      $body.removeClass("is-preload");
    }, 100);
  });

  // ... stopped resizing.
  var resizeTimeout;
  var lastWindowWidth = $window.width();

  $window.on("resize", function () {
    var currentWidth = $window.width();
    // Ignore height-only resize events triggered by mobile URL bar show/hide during scrolling
    if (currentWidth === lastWindowWidth) {
      return;
    }
    lastWindowWidth = currentWidth;

    // Mark as resizing.
    $body.addClass("is-resizing");

    // Unmark after delay.
    clearTimeout(resizeTimeout);

    resizeTimeout = setTimeout(function () {
      $body.removeClass("is-resizing");
    }, 100);
  });

  // Fixes.

  // Object fit images.
  if (!browser.canUse("object-fit") || browser.name == "safari")
    $(".image.object").each(function () {
      var $this = $(this),
        $img = $this.children("img");

      // Hide original image.
      $img.css("opacity", "0");

      // Set background.
      $this
        .css("background-image", 'url("' + $img.attr("src") + '")')
        .css(
          "background-size",
          $img.css("object-fit") ? $img.css("object-fit") : "cover",
        )
        .css(
          "background-position",
          $img.css("object-position") ? $img.css("object-position") : "center",
        );
    });

  // Sidebar.
  var $sidebar = $("#sidebar"),
    $sidebar_inner = $sidebar.children(".inner");

  // Inactive by default on <= large.
  breakpoints.on("<=large", function () {
    $sidebar.addClass("inactive");
  });

  breakpoints.on(">large", function () {
    $sidebar.removeClass("inactive");
  });

  // Hack: Workaround for Chrome/Android scrollbar position bug.
  if (browser.os == "android" && browser.name == "chrome")
    $(
      "<style>#sidebar .inner::-webkit-scrollbar { display: none; }</style>",
    ).appendTo($head);

  // Toggle.
  $('<a href="#sidebar" class="toggle">Toggle</a>')
    .appendTo($sidebar)
    .on("click", function (event) {
      // Prevent default.
      event.preventDefault();
      event.stopPropagation();

      // Toggle.
      $sidebar.toggleClass("inactive");
    });

  // Events.

  // Link clicks inside sidebar navigation (excluding toggle, openers, and in-page anchor links).
  $sidebar.on("click", "a:not(.toggle):not(.opener)", function (event) {
    // >large? Bail.
    if (breakpoints.active(">large")) return;

    // Vars.
    var $a = $(this),
      href = $a.attr("href"),
      target = $a.attr("target");

    // Check URL: ignore empty, hash-only, or anchor links.
    if (!href || href === "#" || href === "" || href.startsWith("#")) return;

    // Prevent default.
    event.preventDefault();
    event.stopPropagation();

    // Hide sidebar.
    $sidebar.addClass("inactive");

    // Redirect to href after transition.
    setTimeout(function () {
      if (target === "_blank") window.open(href);
      else window.location.href = href;
    }, 400);
  });

  // Prevent certain events inside the panel from bubbling.
  $sidebar.on("click touchend touchstart touchmove", function (event) {
    // >large? Bail.
    if (breakpoints.active(">large")) return;

    // Prevent propagation.
    event.stopPropagation();
  });

  // Hide panel on body click/tap.
  $body.on("click touchend", function (event) {
    // >large? Bail.
    if (breakpoints.active(">large")) return;

    // Deactivate.
    $sidebar.addClass("inactive");
  });

  // Scroll lock.

  $window.on("load.sidebar-lock", function () {
    var sh, wh, st;

    // Reset scroll position to 0 if it's 1.
    if ($window.scrollTop() == 1) $window.scrollTop(0);

    $window
      .on("scroll.sidebar-lock", function () {
        var x, y;

        // <=large? Bail without layout thrashing.
        if (breakpoints.active("<=large")) {
          if ($sidebar_inner.data("locked") !== 0) {
            $sidebar_inner.data("locked", 0).css("position", "").css("top", "");
          }
          return;
        }

        // Calculate positions.
        x = Math.max(sh - wh, 0);
        y = Math.max(0, $window.scrollTop() - x);

        // Lock/unlock.
        if ($sidebar_inner.data("locked") == 1) {
          if (y <= 0)
            $sidebar_inner.data("locked", 0).css("position", "").css("top", "");
          else $sidebar_inner.css("top", -1 * x);
        } else {
          if (y > 0)
            $sidebar_inner
              .data("locked", 1)
              .css("position", "fixed")
              .css("top", -1 * x);
        }
      })
      .on("resize.sidebar-lock", function () {
        // Calculate heights.
        wh = $window.height();
        sh = $sidebar_inner.outerHeight() + 30;

        // Trigger scroll.
        $window.trigger("scroll.sidebar-lock");
      })
      .trigger("resize.sidebar-lock");
  });

  // Menu.
  var $menu = $("#menu"),
    $menu_openers = $menu.children("ul").find(".opener");

  // Openers.
  $menu_openers.each(function () {
    var $this = $(this);

    $this.on("click", function (event) {
      // Prevent default.
      event.preventDefault();

      // Toggle.
      $menu_openers.not($this).removeClass("active");
      $this.toggleClass("active");

      // Trigger resize (sidebar lock).
      $window.triggerHandler("resize.sidebar-lock");
    });
  });
})(jQuery);
